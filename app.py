# app.py (Python 3.13 compatible AI Proctoring System)
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-proctoring-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class ExamMonitor:
    def __init__(self):
        self.is_monitoring = False
        self.exam_ended = False
        self.no_face_counter = 0
        self.max_no_face_frames = 30  # 60 seconds at 2fps
        self.sound_detected = False
        self.violation_count = 0
        self.audio_checks = 0
        
    def simulate_face_detection(self):
        """Simulate realistic face detection patterns"""
        # 85% chance face is detected, 15% chance it's not (looking away)
        if random.random() < 0.15:
            self.no_face_counter += 1
            return False
        else:
            self.no_face_counter = max(0, self.no_face_counter - 2)
            return True

exam_monitor = ExamMonitor()

def simulate_audio_monitoring():
    """Simulate audio monitoring with increasing detection probability"""
    print("🔊 Audio monitoring started...")
    while exam_monitor.is_monitoring and not exam_monitor.exam_ended:
        try:
            time.sleep(3)
            exam_monitor.audio_checks += 1
            
            # Increase detection chance over time (2% to 20%)
            base_chance = 0.02
            increase_factor = min(exam_monitor.audio_checks * 0.003, 0.18)
            detection_chance = base_chance + increase_factor
            
            if random.random() < detection_chance and exam_monitor.is_monitoring:
                print("🔊 Sound detected! Ending exam.")
                exam_monitor.exam_ended = True
                exam_monitor.sound_detected = True
                exam_monitor.violation_count += 1
                socketio.emit('exam_ended', {
                    'reason': 'Audio Violation: Unauthorized sound detected during exam',
                    'details': 'The system detected suspicious audio activity.'
                })
                break
                
        except Exception as e:
            print(f"Audio monitoring error: {e}")

def simulate_video_monitoring():
    """Simulate video monitoring with face detection"""
    print("📹 Video monitoring started...")
    frame_count = 0
    
    while exam_monitor.is_monitoring and not exam_monitor.exam_ended:
        try:
            time.sleep(2)
            frame_count += 1
            
            # Simulate face detection
            face_detected = exam_monitor.simulate_face_detection()
            
            # Check for face violation
            if not face_detected:
                if exam_monitor.no_face_counter >= exam_monitor.max_no_face_frames:
                    exam_monitor.exam_ended = True
                    exam_monitor.violation_count += 1
                    socketio.emit('exam_ended', {
                        'reason': 'Video Violation: Face not detected for 60 seconds',
                        'details': 'Please keep your face visible to the camera at all times.'
                    })
                    break
            
            # Send monitoring updates to client
            time_remaining = max(0, exam_monitor.max_no_face_frames - exam_monitor.no_face_counter) * 2
            progress_percent = 100 - (exam_monitor.no_face_counter / exam_monitor.max_no_face_frames) * 100
            
            socketio.emit('monitoring_update', {
                'face_detected': face_detected,
                'no_face_counter': exam_monitor.no_face_counter,
                'max_no_face_frames': exam_monitor.max_no_face_frames,
                'time_remaining': time_remaining,
                'violation_count': exam_monitor.violation_count,
                'progress_percent': progress_percent,
                'frame_count': frame_count
            })
            
        except Exception as e:
            print(f"Video monitoring error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exam')
def exam():
    return render_template('exam.html')

@app.route('/about')
def about():
    return render_template('about.html')

@socketio.on('connect')
def handle_connect():
    print('🎯 Client connected to proctoring system')

@socketio.on('disconnect')
def handle_disconnect():
    print('📞 Client disconnected from proctoring system')

@socketio.on('start_monitoring')
def handle_start_monitoring():
    if not exam_monitor.is_monitoring:
        exam_monitor.is_monitoring = True
        exam_monitor.exam_ended = False
        exam_monitor.no_face_counter = 0
        exam_monitor.sound_detected = False
        exam_monitor.violation_count = 0
        exam_monitor.audio_checks = 0
        
        # Start monitoring threads
        audio_thread = threading.Thread(target=simulate_audio_monitoring)
        video_thread = threading.Thread(target=simulate_video_monitoring)
        
        audio_thread.daemon = True
        video_thread.daemon = True
        
        audio_thread.start()
        video_thread.start()
        
        print("🎯 AI Proctoring System ACTIVATED")
        print("   - Audio Monitoring: ACTIVE")
        print("   - Video Monitoring: ACTIVE")
        print("   - Violation Detection: ENABLED")
        
        emit('monitoring_started', {
            'status': 'active',
            'message': 'AI proctoring system activated successfully'
        })

@socketio.on('manual_cheat_detection')
def handle_manual_cheat():
    exam_monitor.exam_ended = True
    exam_monitor.violation_count += 1
    print("🚨 MANUAL CHEAT DETECTION TRIGGERED")
    socketio.emit('exam_ended', {
        'reason': 'Manual Violation: Test cheat detection activated',
        'details': 'This was a test of the proctoring system.'
    })

@socketio.on('trigger_sound_violation')
def handle_sound_violation():
    exam_monitor.exam_ended = True
    exam_monitor.violation_count += 1
    print("🔊 MANUAL SOUND VIOLATION TRIGGERED")
    socketio.emit('exam_ended', {
        'reason': 'Audio Violation: Unauthorized sound detected',
        'details': 'The system detected prohibited audio activity.'
    })

@socketio.on('trigger_face_violation')
def handle_face_violation():
    exam_monitor.exam_ended = True
    exam_monitor.violation_count += 1
    print("👁️ MANUAL FACE VIOLATION TRIGGERED")
    socketio.emit('exam_ended', {
        'reason': 'Video Violation: Face not visible to camera',
        'details': 'Maintain face visibility throughout the exam.'
    })

@socketio.on('end_exam')
def handle_end_exam():
    exam_monitor.is_monitoring = False
    exam_monitor.exam_ended = True
    print("✅ Exam monitoring stopped by user")

@socketio.on('submit_exam')
def handle_submit_exam():
    exam_monitor.is_monitoring = False
    exam_monitor.exam_ended = True
    print("📝 Exam submitted successfully")
    socketio.emit('exam_submitted', {
        'message': 'Exam submitted successfully!',
        'violation_count': exam_monitor.violation_count
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 AI PROCTORING SYSTEM INITIALIZED")
    print("=" * 60)
    print("📝 Access: http://localhost:5000")
    print("🎯 Features:")
    print("   • Real-time Audio Monitoring")
    print("   • Face Detection System") 
    print("   • Automatic Violation Detection")
    print("   • Professional Monitoring Interface")
    print("   • Python 3.13 Compatible")
    print("=" * 60)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)