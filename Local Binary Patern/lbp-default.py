import cv2
import numpy as np

# Function to compute Local Binary Pattern (LBP)
def compute_lbp(image):
    lbp_image = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            center = image[i, j]
            code = 0
            code |= (image[i-1, j-1] > center) << 7
            code |= (image[i-1, j] > center) << 6
            code |= (image[i-1, j+1] > center) << 5
            code |= (image[i, j+1] > center) << 4
            code |= (image[i+1, j+1] > center) << 3
            code |= (image[i+1, j] > center) << 2
            code |= (image[i+1, j-1] > center) << 1
            code |= (image[i, j-1] > center) << 0
            lbp_image[i, j] = code
    return lbp_image

# Load the pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect faces using Haar cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray_frame[y:y+h, x:x+w]

        # Apply LBP on the face ROI
        lbp_face_roi = compute_lbp(face_roi)

        # Display the face ROI with LBP
        cv2.imshow('LBP Face ROI', lbp_face_roi)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection with LBP', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()