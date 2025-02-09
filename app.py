from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import logging

app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Load YOLO model
model = YOLO("D:/Mini_Project/python_webrtc/weights/best.pt")  # Replace with your YOLO weights path

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['frame']
        logging.debug(f"Received frame of size: {len(file.read())}")
        file.seek(0)
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        results = model.predict(source=frame, conf=0.25)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                label = result.names[int(box.cls)]
                detections.append({
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'label': label,
                    'confidence': float(conf)
                })

        logging.debug(f"Detections: {detections}")
        return jsonify(detections)
    except Exception as e:
        logging.error(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)