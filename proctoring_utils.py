import cv2
import numpy as np
import mediapipe as mp
import pyaudio
import audioop
import threading
import time
import requests

class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)
        
        # Approximate camera matrix (assuming no lens distortion)
        self.focal_length = 950
        self.center = (320, 240)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def get_landmarks(self, face_landmarks, image_shape):
        """Extract specific landmarks for head pose estimation"""
        h, w = image_shape[:2]
        
        # Landmark indices for the model points
        landmark_indices = [1, 152, 33, 263, 61, 291]
        
        image_points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            image_points.append([x, y])
        
        return np.array(image_points, dtype=np.float64)

    def process_frame(self, frame):
        """Process frame to detect head pose"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        head_angles = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get image points
                image_points = self.get_landmarks(face_landmarks, frame.shape)
                
                # Solve PnP to get rotation and translation vectors
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points, image_points, 
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    
                    # Extract Euler angles
                    head_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
                    
                    # Draw axes for visualization
                    nose_end_point2D, _ = cv2.projectPoints(
                        np.array([(0.0, 0.0, 500.0)], dtype=np.float64),
                        rotation_vector, translation_vector, 
                        self.camera_matrix, self.dist_coeffs
                    )
                    
                    p1 = (int(image_points[0][0]), int(image_points[0][1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)
                    
                    # Display angles
                    cv2.putText(frame, f"Yaw: {head_angles[0]:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pitch: {head_angles[1]:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Roll: {head_angles[2]:.1f}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Violations: {self.get_violation_count()}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, head_angles

    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (yaw, pitch, roll)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        x = np.degrees(x)
        y = np.degrees(y)
        z = np.degrees(z)
        
        return (x, y, z)

    def get_violation_count(self):
        # This would typically come from the main app
        # For now, return a placeholder
        return 0


class AudioMonitor:
    def __init__(self, threshold=1000, silence_limit=2, chunk_size=1024, rate=44100):
        self.threshold = threshold
        self.silence_limit = silence_limit
        self.chunk_size = chunk_size
        self.rate = rate
        self.audio = pyaudio.PyAudio()
        self.monitoring = False
        self.silence_start = None
        
    def start_monitoring(self):
        """Start monitoring audio for violations"""
        self.monitoring = True
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("Audio monitoring started...")
        
        while self.monitoring:
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                rms = audioop.rms(data, 2)  # Get RMS value
                
                if rms > self.threshold:
                    print(f"Sound detected! RMS: {rms}")
                    
                    # Send violation to server
                    try:
                        requests.post('http://localhost:5000/audio_violation', timeout=1)
                    except:
                        print("Could not send audio violation to server")
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except OSError as e:
                print(f"Audio stream error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        print("Audio monitoring stopped")
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.monitoring = False