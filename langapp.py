from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from googletrans import Translator
from gtts import gTTS
import cv2
import numpy as np
import uuid
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load YOLO model for sign detection (ensure "best.pt" is in the same directory)
model = YOLO("best.pt")

# Initialize translator for TTS translation
translator = Translator()

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['frame']
        # Read file once and log its size
        file_bytes = file.read()
        logging.debug(f"Received frame of size: {len(file_bytes)} bytes")
        file.seek(0)
        npimg = np.frombuffer(file_bytes, np.uint8)
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

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_text():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data = request.get_json()
        text = data.get("text")
        target_language = data.get("language")
        if not text or not target_language:
            return jsonify({"error": "Missing text or language"}), 400

        # Translate text using googletrans
        translated = translator.translate(text, dest=target_language)

        # Generate a unique filename for the audio file
        filename = f"{uuid.uuid4()}.mp3"
        # Convert the translated text to speech using gTTS and save as MP3
        tts = gTTS(translated.text, lang=target_language)
        tts.save(filename)

        # Return the translated text and an absolute URL for the audio file
        return jsonify({
            "translated_text": translated.text,
            "audio_url": f"http://localhost:5000/audio/{filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    # Serve the MP3 file from the current folder
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
