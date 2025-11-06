from flask import Flask, render_template, request, jsonify, send_from_directory
import os, cv2, numpy as np, base64
from io import BytesIO
from PIL import Image
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

TARGET_PATH = os.path.join('static', 'target_mask.png')

# --- Utilidades de imagen ---
def b64_to_image(b64_string):
    header, encoded = b64_string.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    return arr

def detect_face_and_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Usa el detector Haar (rápido y suficiente para prototipo)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return None
    # si hay varias, toma la más grande
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x,y,w,h) = faces[0]
    pad = int(0.4 * h)  # algo de pad para cubrir máscara que esté abajo/arriba
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad); y2 = min(img_bgr.shape[0], y + h + pad)
    crop = img_bgr[y1:y2, x1:x2]
    return crop

def orb_match_count(imgA, imgB, n_features=1000):
    # convierte a escala de grises y redimensiona para consistencia
    def prep(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h,w = g.shape
        max_sz = 800
        if max(h,w) > max_sz:
            scale = max_sz / float(max(h,w))
            g = cv2.resize(g, (int(w*scale), int(h*scale)))
        return g
    a = prep(imgA); b = prep(imgB)
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return len(good)

# --- Rutas ---
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
    except Exception as e:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

    face_crop = detect_face_and_crop(img)
    if face_crop is None:
        return jsonify({'ok': False, 'msg': 'No face detected'}), 200

    # carga target
    if not os.path.exists(TARGET_PATH):
        return jsonify({'ok': False, 'msg': 'Target image missing on server'}), 500
    target = cv2.imread(TARGET_PATH)
    if target is None:
        return jsonify({'ok': False, 'msg': 'Cannot read target file'}), 500

    matches = orb_match_count(face_crop, target)
    # Umbral: ajustable.  --> empieza con 25 para prototipo; sube si hay falsos positivos.
    THRESH = 25
    recognized = matches >= THRESH

    return jsonify({'ok': True, 'recognized': recognized, 'matches': matches, 'code': "4789#" if recognized else None})

# servir la imagen objetivo (opcional)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # usa $PORT cuando esté en Render, pero para local dejamos 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
