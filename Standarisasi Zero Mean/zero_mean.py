import cv2
import face_recognition
import os
import numpy as np

# Function to load known faces and their names
def load_known_faces(known_faces_dir):
    known_faces = []
    known_names = []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)

    return known_faces, known_names

# Load known faces
known_faces_dir = 'known_faces'
known_faces, known_names = load_known_faces(known_faces_dir)

# Load the pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    face_rects = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces

    # Convert the frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in face_rects]  # Convert to face_recognition format
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Standardize the face region (Zero Mean)
        face_region = gray_frame[top:bottom, left:right]
        face_standardized = (face_region - np.mean(face_region)) / np.std(face_region)

        # Display the standardized face region (optional)
        cv2.imshow('Standardized Face', face_standardized)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
