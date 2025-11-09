# app.py — Face matcher robusto (ORB) para producción
# Coloca las máscaras en: <project_root>/static/targets/
# Coloca codes.json en la raíz del proyecto (map filename -> code)

from flask import Flask, render_template, request, jsonify, send_from_directory
import os, json, cv2, numpy as np, base64, re
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

# ---- RUTAS ABSOLUTAS ----
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR = os.path.join(BASE_DIR, 'static', 'targets')   # aquí van los PNG
CODES_FILE  = os.path.join(BASE_DIR, 'codes.json')

# ---- PARÁMETROS ----
THRESH      = 20      # ajustar si hay muchos falsos positivos/negativos
RATIO_BEST2 = 1.2
# Los códigos que consideras "registrados" (autorizar / mostrar código)
REGISTERED_CODES = {"265", "777", "901"}

os.makedirs(TARGETS_DIR, exist_ok=True)

# ---- ORB / matcher ----
ORB = cv2.ORB_create(nfeatures=1000)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING)

# cache en memoria
TARGETS_CACHE = []

# ---------- utilidades ----------
def b64_to_image(b64_string):
    try:
        header, encoded = b64_string.split(',', 1)
    except Exception:
        encoded = b64_string
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    return np.array(img)[:, :, ::-1]  # RGB -> BGR

def detect_face_and_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x,y,w,h = faces[0]
    pad = int(0.35 * h)
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

def load_codes():
    if os.path.exists(CODES_FILE):
        with open(CODES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def infer_code_from_name(filename):
    digits = ''.join(re.findall(r'\d+', filename))
    return digits if digits else filename

def preload_targets():
    """Carga imágenes desde TARGETS_DIR y precalcula kp/des para ORB."""
    global TARGETS_CACHE
    TARGETS_CACHE = []
    codes_map = load_codes()
    if not os.path.exists(TARGETS_DIR):
        return 0
    for fname in sorted(os.listdir(TARGETS_DIR)):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.webp')):
            continue
        path = os.path.join(TARGETS_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        g = prep_gray_resized(img)
        kp, des = ORB.detectAndCompute(g, None)
        code = str(codes_map.get(fname, infer_code_from_name(fname)))
        TARGETS_CACHE.append({"filename": fname, "code": code, "kp": kp, "des": des})
    return len(TARGETS_CACHE)

def ensure_targets_loaded():
    if not TARGETS_CACHE:
        n = preload_targets()
        if n == 0:
            raise RuntimeError("No targets found in static/targets. Sube tus PNG y codes.json")

def orb_match_count_des(des1, des2):
    if des1 is None or des2 is None:
        return 0
    try:
        matches = BF.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    good = 0
    for m_n in matches:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good += 1
    return good

# ---------- rutas ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    """
    Escaneo modo juego — cualquier rostro activa el mensaje
    """
    try:
        # Recibe la imagen pero no hace validación (modo libre)
        data = request.json.get('image')
        if not data:
            return jsonify({'ok': False, 'msg': 'No image sent'}), 400

        # Solo decodifica para evitar error de formato
        try:
            _ = base64.b64decode(data.split(',', 1)[1])
        except Exception:
            return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

        # Modo juego: siempre muestra el mensaje sin validar
        return jsonify({
            'ok': True,
            'recognized': True,
            'code': '007',  # puedes poner 265, 777 o el que quieras mostrar
            'matches': 0,
            'second_best': 0,
            'message': "agente encubierto, puedes pasar al bunker con tu equipo"
        }), 200

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/reload_targets', methods=['POST'])
def reload_targets():
    n = preload_targets()
    return jsonify({'ok': True, 'loaded': n})

@app.route('/add_target', methods=['POST'])
def add_target():
    payload = request.get_json(silent=True) or {}
    b64 = payload.get('image'); fname = payload.get('filename'); code = payload.get('code')
    if not (b64 and fname and code):
        return jsonify({'ok': False, 'msg': 'image, filename, code required'}), 400
    try:
        img = b64_to_image(b64)
    except Exception:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400
    save_path = os.path.join(TARGETS_DIR, fname)
    ok = cv2.imwrite(save_path, img)
    if not ok:
        return jsonify({'ok': False, 'msg': 'Cannot write image'}), 500
    codes = load_codes(); codes[fname] = str(code)
    with open(CODES_FILE, 'w', encoding='utf-8') as f:
        json.dump(codes, f, ensure_ascii=False, indent=2)
    preload_targets()
    return jsonify({'ok': True, 'saved': fname, 'code': str(code)}), 200

@app.route('/diag')
def diag():
    try:
        ensure_targets_loaded()
        stats = []
        for t in TARGETS_CACHE:
            stats.append({
                'file': t['filename'],
                'code': t['code'],
                'kp': 0 if t['kp'] is None else len(t['kp']),
                'des': 0 if t['des'] is None else len(t['des'])
            })
        return jsonify({'ok': True, 'targets': stats, 'thresh': THRESH, 'ratio': RATIO_BEST2}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename)

if __name__ == '__main__':
    preload_targets()  # carga inicial
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
