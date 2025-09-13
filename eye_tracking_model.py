import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist

class EyeTrackingModel:
    def __init__(self):
        # Initialize dlib's face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Define the indexes for the left and right eye
        self.LEFT_EYE_START, self.LEFT_EYE_END = 42, 48
        self.RIGHT_EYE_START, self.RIGHT_EYE_END = 36, 42
        
    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def detect_eyes(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        rects = self.detector(gray, 0)
        
        # Initialize eye aspects ratios
        left_ear = 0
        right_ear = 0
        
        # Loop over the face detections
        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = self.predictor(gray, rect)
            shape = self.shape_to_np(shape)
            
            # Extract the left and right eye coordinates
            left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            # Calculate the eye aspect ratio for both eyes
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio together for both eyes
            ear = (left_ear + right_ear) / 2.0
            
            return ear, left_eye, right_eye
        
        return 0, None, None
    
    def shape_to_np(self, shape, dtype="int"):
        # Initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        
        # Loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        
        return coords
    
    def predict_eye_deviation(self, image):
        # Detect eyes and calculate eye aspect ratio
        ear, left_eye, right_eye = self.detect_eyes(image)
        
        if ear == 0:
            return 0  # No face detected
        
        # Calculate deviation from normal eye aspect ratio (0.25 is typical for open eyes)
        deviation = abs(ear - 0.25) * 400  # Scale to percentage
        
        return min(deviation, 100)
    
    def predict_head_position(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        rects = self.detector(gray, 0)
        
        if len(rects) == 0:
            return 0  # No face detected
        
        # Get the first face
        rect = rects[0]
        
        # Calculate head position based on face position in the frame
        frame_center_x = image.shape[1] / 2
        face_center_x = rect.center().x
        
        # Calculate deviation from center
        deviation = abs(face_center_x - frame_center_x) / frame_center_x * 100
        
        return min(deviation, 100)

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = EyeTrackingModel()
    
    # Load a test image
    image = cv2.imread("test_face.jpg")
    
    if image is not None:
        # Predict eye deviation and head position
        eye_deviation = model.predict_eye_deviation(image)
        head_position = model.predict_head_position(image)
        
        print(f"Eye Deviation: {eye_deviation:.2f}%")
        print(f"Head Position: {head_position:.2f}%")
    else:
        print("Test image not found")