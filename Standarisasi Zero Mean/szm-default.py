import cv2
import face_recognition
import numpy as np

# Load the pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Crop the face region
        face_region = gray_frame[y:y+h, x:x+w]

        # Resize the face region to 128x128 (optional, for better recognition)
        face_resized = cv2.resize(face_region, (128, 128))

        # Standardize the face region (Zero Mean)
        face_standardized = (face_resized - np.mean(face_resized)) / np.std(face_resized)

        # Display the standardized face region
        cv2.imshow('Standardized Face', face_standardized)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
