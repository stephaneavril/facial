from flask import Flask, render_template, request, jsonify, send_from_directory
import os, json, cv2, numpy as np, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

# --- Rutas y config ---
TARGETS_DIR = os.path.join('static', 'targets')
CODES_FILE  = 'codes.json'
THRESH      = 25          # umbral mínimo de matches "buenos"
RATIO_BEST2 = 1.25        # mejor debe superar al segundo al menos 25% (evita falsos positivos)

os.makedirs(TARGETS_DIR, exist_ok=True)

# --- Utilidades ---
def b64_to_image(b64_string):
    """
    Convierte 'data:image/...;base64,XXXX' -> ndarray BGR (OpenCV)
    """
    header, encoded = b64_string.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    return arr

def detect_face_and_crop(img_bgr):
    """
    Detecta rostro y hace un recorte con padding para cubrir máscara
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x,y,w,h) = faces[0]
    pad = int(0.4 * h)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad); y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]

def prep_gray_resized(im_bgr, max_sz=800):
    g = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    h,w = g.shape
    if max(h,w) > max_sz:
        scale = max_sz / float(max(h,w))
        g = cv2.resize(g, (int(w*scale), int(h*scale)))
    return g

# --- Índice de targets en memoria (cache ORB) ---
ORB = cv2.ORB_create(nfeatures=1000)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING)

TARGETS_CACHE = []  # cada item: {"filename","code","img","kp","des"}

def load_codes():
    if os.path.exists(CODES_FILE):
        with open(CODES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def infer_code_from_name(filename):
    # si no hay codes.json para ese archivo: usa dígitos del nombre, si no hay, usa nombre completo
    import re
    digits = ''.join(re.findall(r'\d+', filename))
    return digits if digits else filename

def preload_targets():
    """
    Carga todas las imágenes en static/targets y precalcula kp/des.
    """
    global TARGETS_CACHE
    TARGETS_CACHE = []

    codes_map = load_codes()

    for fname in sorted(os.listdir(TARGETS_DIR)):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','webp')):
            continue
        path = os.path.join(TARGETS_DIR, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        g = prep_gray_resized(img)
        kp, des = ORB.detectAndCompute(g, None)
        code = codes_map.get(fname, infer_code_from_name(fname))
        TARGETS_CACHE.append({
            "filename": fname,
            "code": code,
            "img": img,
            "kp": kp,
            "des": des
        })
    return len(TARGETS_CACHE)

# Carga inicial
preload_targets()

def orb_match_count_des(face_des, target_des):
    if face_des is None or target_des is None:
        return 0
    try:
        matches = BF.knnMatch(face_des, target_des, k=2)
    except cv2.error:
        return 0
    good = 0
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good += 1
    return good

# --- Vistas ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    data = request.json.get('image')
    if not data:
        return jsonify({'ok': False, 'msg': 'No image sent'}), 400

    try:
        img = b64_to_image(data)
    except Exception:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

    face = detect_face_and_crop(img)
    if face is None:
        return jsonify({'ok': False, 'msg': 'No face detected'}), 200

    g = prep_gray_resized(face)
    kp, des = ORB.detectAndCompute(g, None)

    # compara contra todos los targets
    scores = []
    for t in TARGETS_CACHE:
        good = orb_match_count_des(des, t["des"])
        scores.append((good, t))

    if not scores:
        return jsonify({'ok': False, 'msg': 'No targets loaded'}), 500

    scores.sort(key=lambda x: x[0], reverse=True)
    best_good, best_t = scores[0]
    second_good = scores[1][0] if len(scores) > 1 else 0

    recognized = (best_good >= THRESH) and (best_good >= second_good * RATIO_BEST2)

    return jsonify({
        'ok': True,
        'recognized': recognized,
        'matches': int(best_good),
        'second_best': int(second_good),
        'filename': best_t["filename"] if recognized else None,
        'code': best_t["code"] if recognized else None
    })

@app.route('/reload_targets', methods=['POST'])
def reload_targets():
    n = preload_targets()
    return jsonify({'ok': True, 'loaded': n})

@app.route('/add_target', methods=['POST'])
def add_target():
    """
    Opcional: subir una nueva máscara como base64 y registrar su código.
    JSON:
    {
      "image": "data:image/png;base64,...",
      "filename": "mask_4.png",
      "code": "123"
    }
    """
    payload = request.get_json(silent=True) or {}
    b64 = payload.get('image')
    fname = payload.get('filename')
    code  = payload.get('code')

    if not (b64 and fname and code):
        return jsonify({'ok': False, 'msg': 'image, filename, code are required'}), 400

    # guardar imagen
    try:
        img = b64_to_image(b64)
    except Exception:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

    save_path = os.path.join(TARGETS_DIR, fname)
    cv2.imwrite(save_path, img)

    # actualizar codes.json
    codes = load_codes()
    codes[fname] = str(code)
    with open(CODES_FILE, 'w', encoding='utf-8') as f:
        json.dump(codes, f, ensure_ascii=False, indent=2)

    # recargar cache
    preload_targets()
    return jsonify({'ok': True, 'saved': fname, 'code': str(code)})

# Servir estáticos (opcional extra)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    