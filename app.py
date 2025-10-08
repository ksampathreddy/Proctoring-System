# app.py (Python 3.13 compatible - No eventlet)
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
# Use threading instead of eventlet for Python 3.13 compatibility
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class ExamMonitor:
    def __init__(self):
        self.is_monitoring = False
        self.exam_ended = False
        self.no_face_counter = 0
        self.max_no_face_frames = 30
        self.sound_detected = False
        self.violation_count = 0
        
    def simulate_face_detection(self):
        """Simulate face detection based on random patterns"""
        # Simulate realistic face detection patterns
        # 85% chance face is detected, 15% chance it's not (looking away)
        if random.random() < 0.15:  # 15% chance of no face
            self.no_face_counter += 1
            return False
        else:
            self.no_face_counter = max(0, self.no_face_counter - 2)
            return True

exam_monitor = ExamMonitor()

def simulate_audio_monitoring():
    """Simulate audio monitoring"""
    print("🔊 Audio monitoring started...")
    check_count = 0
    while exam_monitor.is_monitoring and not exam_monitor.exam_ended:
        try:
            time.sleep(3)  # Check every 3 seconds
            check_count += 1
            
            # Simulate random sound detection (increases over time)
            detection_chance = 0.02 + (check_count * 0.005)  # Increases from 2% to ~17%
            if random.random() < min(detection_chance, 0.17) and exam_monitor.is_monitoring:
                print("🔊 Sound detected! Ending exam.")
                exam_monitor.exam_ended = True
                exam_monitor.sound_detected = True
                exam_monitor.violation_count += 1
                socketio.emit('exam_ended', {'reason': 'Audio violation: Unauthorized sound detected'})
                break
                
        except Exception as e:
            print(f"Audio monitoring error: {e}")

def simulate_video_monitoring():
    """Simulate video monitoring"""
    print("📹 Video monitoring started...")
    while exam_monitor.is_monitoring and not exam_monitor.exam_ended:
        try:
            time.sleep(2)  # Update every 2 seconds
            
            # Simulate face detection
            face_detected = exam_monitor.simulate_face_detection()
            
            # Check if no face detected for too long
            if not face_detected:
                if exam_monitor.no_face_counter >= exam_monitor.max_no_face_frames:
                    exam_monitor.exam_ended = True
                    exam_monitor.violation_count += 1
                    socketio.emit('exam_ended', {
                        'reason': 'Video violation: Face not detected for 60 seconds'
                    })
                    break
            
            # Send monitoring data to client
            socketio.emit('monitoring_update', {
                'face_detected': face_detected,
                'no_face_counter': exam_monitor.no_face_counter,
                'max_no_face_frames': exam_monitor.max_no_face_frames,
                'time_remaining': max(0, exam_monitor.max_no_face_frames - exam_monitor.no_face_counter) * 2,
                'violation_count': exam_monitor.violation_count
            })
            
        except Exception as e:
            print(f"Video monitoring error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exam')
def exam():
    return render_template('exam.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_monitoring')
def handle_start_monitoring():
    if not exam_monitor.is_monitoring:
        exam_monitor.is_monitoring = True
        exam_monitor.exam_ended = False
        exam_monitor.no_face_counter = 0
        exam_monitor.sound_detected = False
        exam_monitor.violation_count = 0
        
        # Start simulated monitoring threads
        audio_thread = threading.Thread(target=simulate_audio_monitoring)
        video_thread = threading.Thread(target=simulate_video_monitoring)
        
        audio_thread.daemon = True
        video_thread.daemon = True
        
        audio_thread.start()
        video_thread.start()
        
        print("🎯 AI Proctoring started - Audio and Video monitoring active")
        emit('monitoring_started', {'status': 'active'})

@socketio.on('manual_cheat_detection')
def handle_manual_cheat():
    """Allow manual cheat detection for testing"""
    exam_monitor.exam_ended = True
    exam_monitor.violation_count += 1
    print("🚨 Manual cheat detection triggered")
    socketio.emit('exam_ended', {'reason': 'Manual cheat detection triggered'})

@socketio.on('trigger_sound_violation')
def handle_sound_violation():
    """Trigger sound violation for testing"""
    exam_monitor.exam_ended = True
    exam_monitor.violation_count += 1
    print("🔊 Manual sound violation triggered")
    socketio.emit('exam_ended', {'reason': 'Sound violation: Unauthorized audio detected'})

@socketio.on('trigger_face_violation')
def handle_face_violation():
    """Trigger face violation for testing"""
    exam_monitor.exam_ended = True
    exam_monitor.violation_count += 1
    print("👁️ Manual face violation triggered")
    socketio.emit('exam_ended', {'reason': 'Face violation: Candidate not visible'})

@socketio.on('end_exam')
def handle_end_exam():
    exam_monitor.is_monitoring = False
    exam_monitor.exam_ended = True
    print("✅ Monitoring stopped")

if __name__ == '__main__':
    print("🚀 Starting AI Proctoring System...")
    print("📝 Access the application at: http://localhost:5000")
    print("🎯 Features:")
    print("   - Real-time audio monitoring (simulated)")
    print("   - Face detection monitoring (simulated)") 
    print("   - Automatic violation detection")
    print("   - Professional monitoring panel")
    print("   - Python 3.13 compatible")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)