from flask import Flask, request, jsonify, render_template
import cv2
import io
import base64
import numpy as np
from ultralytics import YOLO

from model_pred import get_pred
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect/', methods=['GET', 'POST'])
def detect():
    model = YOLO('models/cloth_detection.pt')
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No image provided')

        image_file = request.files['image']
        try:
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            pimg = get_pred(model, image)
            _, buffer = cv2.imencode('.jpg', pimg)
            encoded_img = base64.b64encode(buffer).decode('utf-8')
            img_data = f'data:image/jpeg;base64,{encoded_img}'
            
            return render_template('index.html', image=img_data)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

@app.route('/health/', methods=['GET'])
def health():
    return render_template('health.html', status="The API is healthy")

if __name__ == '__main__':
    app.run(debug=True)
