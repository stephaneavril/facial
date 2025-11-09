# app.py — Reconocimiento multi-máscara con mensaje fijo estilo “James Bond”
from flask import Flask, render_template, request, jsonify, send_from_directory
import os, json, cv2, numpy as np, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

# --- Rutas absolutas ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TARGETS_DIR = os.path.join(BASE_DIR, 'static', 'targets')   # donde están tus PNG
CODES_FILE  = os.path.join(BASE_DIR, 'codes.json')

# --- Parámetros y lista blanca ---
THRESH      = 20
RATIO_BEST2 = 1.20
MESSAGE_OK  = "agente encubierto, puedes pasar al bunker con tu equipo"
WHITELIST_CODES = {"265", "777", "901"}   # cualquiera de estas autoriza

os.makedirs(TARGETS_DIR, exist_ok=True)
ORB = cv2.ORB_create(nfeatures=1000)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING)
TARGETS_CACHE = []

# ---------- Utilidades ----------
def b64_to_image(b64_string):
    header, encoded = b64_string.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    return np.array(img)[:, :, ::-1]

def detect_face_and_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0: return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x,y,w,h) = faces[0]
    pad = int(0.4*h)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad); y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]

def prep_gray_resized(im, max_sz=800):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h,w = g.shape
    if max(h,w) > max_sz:
        scale = max_sz / float(max(h,w))
        g = cv2.resize(g, (int(w*scale), int(h*scale)))
    return g

def load_codes():
    if os.path.exists(CODES_FILE):
        with open(CODES_FILE,'r',encoding='utf-8') as f: return json.load(f)
    return {}

def infer_code_from_name(fname):
    import re
    digits = ''.join(re.findall(r'\d+', fname))
    return digits if digits else fname

def preload_targets():
    global TARGETS_CACHE
    TARGETS_CACHE = []
    codes = load_codes()
    for fname in sorted(os.listdir(TARGETS_DIR)):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.webp')): continue
        path = os.path.join(TARGETS_DIR, fname)
        img = cv2.imread(path)
        if img is None: continue
        g = prep_gray_resized(img)
        kp, des = ORB.detectAndCompute(g, None)
        code = str(codes.get(fname, infer_code_from_name(fname)))
        TARGETS_CACHE.append({"filename": fname, "code": code, "kp": kp, "des": des})
    return len(TARGETS_CACHE)

def ensure_targets_loaded():
    if not TARGETS_CACHE:
        n = preload_targets()
        if n == 0: raise RuntimeError("No targets encontrados en static/targets")

def orb_match_count_des(des1, des2):
    if des1 is None or des2 is None: return 0
    try: matches = BF.knnMatch(des1, des2, k=2)
    except cv2.error: return 0
    good=0
    for m_n in matches:
        if len(m_n)!=2: continue
        m,n = m_n
        if m.distance < 0.75*n.distance: good+=1
    return good

# ---------- Rutas ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    try:
        ensure_targets_loaded()
        data = request.json.get('image')
        if not data: return jsonify({'ok':False,'msg':'No image sent'}),400
        img = b64_to_image(data)
        face = detect_face_and_crop(img)
        if face is None: face = img
        g = prep_gray_resized(face)
        kp, des = ORB.detectAndCompute(g, None)
        if des is None or len(des)==0:
            return jsonify({'ok':True,'recognized':False,'msg':'No features'}),200

        best_good, second_good, best_t = -1, 0, None
        for t in TARGETS_CACHE:
            good = orb_match_count_des(des, t["des"])
            if good > best_good:
                second_good = best_good if best_good>=0 else 0
                best_good = good; best_t = t
            elif good > second_good: second_good = good
        if best_t is None:
            return jsonify({'ok':True,'recognized':False,'msg':'No valid targets'}),200

        recognized = (best_good>=THRESH) and (best_good>=second_good*RATIO_BEST2)
        authorized = recognized and (best_t["code"] in WHITELIST_CODES)

        return jsonify({
            'ok':True,
            'recognized':authorized,
            'matches':int(best_good),
            'second_best':int(second_good),
            'code':best_t["code"] if authorized else None,
            'message':MESSAGE_OK if authorized else None
        }),200
    except Exception as e:
        return jsonify({'ok':False,'error':str(e)}),500

@app.route('/diag')
def diag():
    try:
        ensure_targets_loaded()
        stats=[{"file":t["filename"],"code":t["code"],
                "kp":0 if t["kp"] is None else len(t["kp"]),
                "des":0 if t["des"] is None else len(t["des"])} for t in TARGETS_CACHE]
        return jsonify({"ok":True,"targets":stats})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=False)
