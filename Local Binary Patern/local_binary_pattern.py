import cv2
import numpy as np
import os

# Function to compute Local Binary Pattern (LBP)
def compute_lbp(image):
    lbp_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            code = 0
            code |= (image[i - 1, j - 1] > center) << 7
            code |= (image[i - 1, j] > center) << 6
            code |= (image[i - 1, j + 1] > center) << 5
            code |= (image[i, j + 1] > center) << 4
            code |= (image[i + 1, j + 1] > center) << 3
            code |= (image[i + 1, j] > center) << 2
            code |= (image[i + 1, j - 1] > center) << 1
            code |= (image[i, j - 1] > center) << 0
            lbp_image[i, j] = code
    return lbp_image

# Function to compute histogram of LBP image
def compute_histogram(lbp_image):
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(257), range=(0, 256))
    hist = hist.astype("float32")  # Ensure the histogram is in float32 format
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

# Load known faces from images in 'known_faces' directory
known_faces = {}
known_histograms = {}
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
known_faces_dir = os.path.join(current_dir, 'known_faces')

valid_extensions = ('.jpg', '.jpeg', '.png')  # Valid image extensions

for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_dir):
        person_histograms = []
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            if os.path.isfile(img_path) and img_path.lower().endswith(valid_extensions):
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                lbp_face = compute_lbp(image)
                hist = compute_histogram(lbp_face)
                person_histograms.append(hist)
        if person_histograms:
            known_histograms[person_name] = person_histograms

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
        face_roi = gray_frame[y:y + h, x:x + w]

        # Apply LBP on the face ROI
        lbp_face_roi = compute_lbp(face_roi)
        hist_face_roi = compute_histogram(lbp_face_roi)

        # Compare with known faces
        best_match_name = "Unknown"
        best_match_score = float("inf")

        for name, histograms in known_histograms.items():
            for known_hist in histograms:
                # Compute chi-squared distance between histograms
                score = cv2.compareHist(hist_face_roi, known_hist, cv2.HISTCMP_CHISQR)
                if score < best_match_score:
                    best_match_score = score
                    best_match_name = name

        if best_match_score < 0.5:  # Adjust the threshold based on your needs
            cv2.putText(frame, best_match_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the face ROI with LBP
        cv2.imshow('LBP Face ROI', lbp_face_roi)

    # Display the resulting frame
    cv2.imshow('Face Detection with LBP', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
