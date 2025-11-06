from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'png','jpg','jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'change-this-secret'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def is_match(upload_path, target_path, min_matches=25):
    """Compare two images using ORB feature matching. Returns True if enough good matches."""
    img1 = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(upload_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return False, 0
    # resize for speed if too large
    max_size = 800
    def resize_if_needed(img):
        h,w = img.shape
        if max(h,w) > max_size:
            scale = max_size / float(max(h,w))
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        return img
    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return False, 0
    # BF matcher with Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    # ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return (len(good) >= min_matches), len(good)

@app.route('/', methods=['GET','POST'])
def index():
    code = None
    votes = None
    if request.method == 'POST':
        if 'photo' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['photo']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            target_path = os.path.join('static', 'target_mask.png')
            match, n_matches = is_match(upload_path, target_path)
            if match:
                # Detected target mask -> reveal code
                code = "4789#"
            else:
                flash(f"No match detected (good matches: {n_matches}). Try another photo or adjust lighting.")
    return render_template('index.html', code=code)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
