from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simulated ML model for eye tracking
class EyeTrackingModel:
    def __init__(self):
        # In a real implementation, this would load a trained model
        pass
    
    def predict_eye_deviation(self, image_data):
        # Simulate ML model prediction
        # In a real implementation, this would process the image and return eye deviation
        deviation = np.random.uniform(0, 100)
        return deviation
    
    def predict_head_position(self, image_data):
        # Simulate ML model prediction
        # In a real implementation, this would process the image and return head position
        position = np.random.uniform(0, 100)
        return position

# Initialize the model
model = EyeTrackingModel()

@app.route('/api/analyze_eye_movement', methods=['POST'])
def analyze_eye_movement():
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Convert base64 image to numpy array
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Simulate processing time
        import time
        time.sleep(0.1)
        
        # Get predictions from model
        eye_deviation = model.predict_eye_deviation(image)
        head_position = model.predict_head_position(image)
        
        return jsonify({
            'eye_deviation': round(eye_deviation, 2),
            'head_position': round(head_position, 2),
            'gaze_stability': round(100 - eye_deviation - head_position/2, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)