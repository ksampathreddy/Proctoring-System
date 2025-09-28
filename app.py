from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import pyaudio
import wave
import audioop
from proctoring_utils import HeadPoseDetector, AudioMonitor

app = Flask(__name__)

# Global variables
head_pose_detector = HeadPoseDetector()
audio_monitor = AudioMonitor()
exam_active = True
violation_count = 0
MAX_VIOLATIONS = 3

# Video streaming generator
def generate_frames():
    global exam_active, violation_count
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while exam_active and violation_count < MAX_VIOLATIONS:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame for head pose detection
        processed_frame, head_angles = head_pose_detector.process_frame(frame)
        
        # Check for head rotation violations
        if head_angles:
            yaw, pitch, roll = head_angles
            
            # Check if head rotation exceeds threshold (in degrees)
            if abs(yaw) > 30 or abs(pitch) > 25:
                violation_count += 1
                print(f"Head rotation violation detected! Count: {violation_count}")
                
                # If max violations reached, end exam
                if violation_count >= MAX_VIOLATIONS:
                    exam_active = False
                    print("Exam terminated due to excessive head movement")
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_exam')
def start_exam():
    global exam_active, violation_count
    exam_active = True
    violation_count = 0
    
    # Start audio monitoring in a separate thread
    audio_thread = threading.Thread(target=audio_monitor.start_monitoring)
    audio_thread.daemon = True
    audio_thread.start()
    
    return jsonify({"status": "Exam started"})

@app.route('/end_exam')
def end_exam():
    global exam_active
    exam_active = False
    audio_monitor.stop_monitoring()
    return jsonify({"status": "Exam ended"})

@app.route('/exam_status')
def exam_status():
    global exam_active, violation_count
    return jsonify({
        "active": exam_active,
        "violations": violation_count,
        "max_violations": MAX_VIOLATIONS
    })

@app.route('/audio_violation', methods=['POST'])
def audio_violation():
    global exam_active, violation_count
    
    violation_count += 1
    print(f"Audio violation detected! Count: {violation_count}")
    
    if violation_count >= MAX_VIOLATIONS:
        exam_active = False
        return jsonify({"status": "exam_ended", "reason": "audio"})
    
    return jsonify({"status": "violation_recorded"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)