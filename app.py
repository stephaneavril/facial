# app.py — Reconocimiento de máscaras múltiples con ORB
# -----------------------------------------------
# Coloca tus PNG de máscaras en: static/targets/
# Define los códigos en codes.json (archivo -> código).
# Ejemplo codes.json:
# {
#   "mask_265.png": "265",
#   "mask_901.png": "901",
#   "mask_777.png": "777"
# }

from flask import Flask, render_template, request, jsonify, send_from_directory
import os, json, cv2, numpy as np, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB

# --- Configuración ---
TARGETS_DIR = os.path.join('static', 'targets')
CODES_FILE  = 'codes.json'

# Umbrales (ajusta según pruebas)
THRESH      = 16       # mínimo de matches "buenos" para reconocer (empezar bajo para demo)
RATIO_BEST2 = 1.15     # mejor debe superar al segundo al menos 15%

os.makedirs(TARGETS_DIR, exist_ok=True)

# --- ORB global + matcher ---
ORB = cv2.ORB_create(nfeatures=1000)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING)

# Cache en memoria: cada item: {"filename","code","kp","des"}
TARGETS_CACHE = []

# ---------- Utilidades de imagen ----------
def b64_to_image(b64_string):
    """Convierte 'data:image/...;base64,xxxx' -> ndarray BGR (OpenCV)"""
    header, encoded = b64_string.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    return arr

def detect_face_and_crop(img_bgr):
    """Detecta rostro y recorta con padding; si no detecta, retorna None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return None
    # toma la cara más grande
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x,y,w,h) = faces[0]
    pad = int(0.40 * h)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad); y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]

def prep_gray_resized(im_bgr, max_sz=800):
    """Grises + resize para consistencia."""
    g = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    h,w = g.shape
    if max(h,w) > max_sz:
        scale = max_sz / float(max(h,w))
        g = cv2.resize(g, (int(w*scale), int(h*scale)))
    return g

# ---------- Carga de códigos/targets ----------
def load_codes():
    if os.path.exists(CODES_FILE):
        with open(CODES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def infer_code_from_name(filename):
    """Si no hay entrada en codes.json, infiere con dígitos del nombre."""
    import re
    digits = ''.join(re.findall(r'\d+', filename))
    return digits if digits else filename

def preload_targets():
    """Carga todos los PNG/JPG en static/targets y precalcula descriptores ORB."""
    global TARGETS_CACHE
    TARGETS_CACHE = []

    codes_map = load_codes()
    files = sorted(os.listdir(TARGETS_DIR))
    for fname in files:
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
        path = os.path.join(TARGETS_DIR, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        g = prep_gray_resized(img)
        kp, des = ORB.detectAndCompute(g, None)
        code = str(codes_map.get(fname, infer_code_from_name(fname)))
        TARGETS_CACHE.append({
            "filename": fname,
            "code": code,
            "kp": kp,
            "des": des
        })
    return len(TARGETS_CACHE)

def ensure_targets_loaded():
    """Asegura que el índice esté cargado; si no, recarga."""
    if not TARGETS_CACHE:
        n = preload_targets()
        if n == 0:
            raise RuntimeError("No targets found in static/targets. ¿Subiste los PNG y codes.json?")

# ---------- Matching ----------
def orb_match_count_des(face_des, target_des):
    """Cuenta matches 'buenos' entre descriptores ya calculados."""
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

# ---------- Rutas ----------
@app.route('/')
def index():
    # Si no tienes template, puedes devolver un texto simple o servir tu index.html
    try:
        return render_template('index.html')
    except Exception:
        return "Face Matcher live. Usa POST /scan", 200

@app.route('/scan', methods=['POST'])
def scan():
    """Recibe JSON {'image': 'data:image/...;base64,xxx'} y devuelve reconocimiento."""
    try:
        ensure_targets_loaded()

        data = request.json.get('image')
        if not data:
            return jsonify({'ok': False, 'msg': 'No image sent'}), 400

        # 1) Decodificar
        try:
            img = b64_to_image(data)
        except Exception:
            return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

        # 2) Intentar rostro; si falla, usar imagen completa (útil con medias)
        face = detect_face_and_crop(img)
        used_fallback = False
        if face is None:
            face = img
            used_fallback = True

        g = prep_gray_resized(face)
        kp, des = ORB.detectAndCompute(g, None)
        if des is None or len(des) == 0:
            return jsonify({'ok': True, 'recognized': False, 'msg': 'No features on image', 'fallback': used_fallback}), 200

        # 3) Comparar contra todas las máscaras
        best_good = -1
        second_good = 0
        best_t = None
        for t in TARGETS_CACHE:
            good = orb_match_count_des(des, t["des"])
            if good > best_good:
                second_good = best_good if best_good >= 0 else 0
                best_good = good
                best_t = t
            elif good > second_good:
                second_good = good

        if best_t is None:
            return jsonify({'ok': True, 'recognized': False, 'msg': 'No valid targets'}), 200

        recognized = (best_good >= THRESH) and (best_good >= second_good * RATIO_BEST2)

        return jsonify({
            'ok': True,
            'recognized': recognized,
            'matches': int(best_good),
            'second_best': int(second_good),
            'filename': best_t["filename"] if recognized else None,
            'code': best_t["code"] if recognized else None,
            'fallback': used_fallback
        }), 200

    except Exception as e:
        # Evita 500 silenciosos: devuelve detalle para depurar
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/reload_targets', methods=['POST'])
def reload_targets():
    """Recarga los PNG del servidor (útil tras subir nuevas máscaras)."""
    n = preload_targets()
    return jsonify({'ok': True, 'loaded': n})

@app.route('/add_target', methods=['POST'])
def add_target():
    """
    Sube una nueva máscara por API.
    JSON:
    {
      "image": "data:image/png;base64,...",
      "filename": "mask_123.png",
      "code": "123"
    }
    """
    payload = request.get_json(silent=True) or {}
    b64 = payload.get('image')
    fname = payload.get('filename')
    code  = payload.get('code')

    if not (b64 and fname and code):
        return jsonify({'ok': False, 'msg': 'image, filename, code are required'}), 400

    # Guardar imagen
    try:
        img = b64_to_image(b64)
    except Exception:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

    save_path = os.path.join(TARGETS_DIR, fname)
    ok = cv2.imwrite(save_path, img)
    if not ok:
        return jsonify({'ok': False, 'msg': 'Cannot write image'}), 500

    # Actualizar codes.json
    codes = load_codes()
    codes[fname] = str(code)
    with open(CODES_FILE, 'w', encoding='utf-8') as f:
        json.dump(codes, f, ensure_ascii=False, indent=2)

    # Recargar cache
    preload_targets()
    return jsonify({'ok': True, 'saved': fname, 'code': str(code)})

@app.route('/diag')
def diag():
    """Diagnóstico: cuántos targets cargados y si tienen descriptores."""
    try:
        ensure_targets_loaded()
        stats = []
        for t in TARGETS_CACHE:
            kp_len = 0 if t["kp"] is None else len(t["kp"])
            des_len = 0 if t["des"] is None else len(t["des"])
            stats.append({"file": t["filename"], "code": t["code"], "kp": kp_len, "des": des_len})
        return jsonify({"ok": True, "threshold": THRESH, "ratio_best2": RATIO_BEST2, "targets": stats})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Servir estáticos (si lo necesitas explícito)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ---------- Main ----------
if __name__ == '__main__':
    # Para local: python app.py (Render usará gunicorn)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    