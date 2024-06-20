import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'

# All the faces data will be present in this folder
datasets = 'datasets'

# These are sub datasets of the folder,
# for my face, I've used my name. You can change the label here
sub_data = 'Sye'

# Create the parent directory if it doesn't exist
if not os.path.isdir(datasets):
    os.mkdir(datasets)

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# Define the size of images
(width, height) = (130, 100)

# '0' is used for my webcam,
# if you've any other camera attached use '1' like this
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# The program loops until it has 30 images of the face.
count = 1
while count <= 30:  # Loop until 30 images are captured
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1  # Increment count inside the for loop to ensure only face images are counted
    
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # Exit on ESC key
        break

webcam.release()
cv2.destroyAllWindows()