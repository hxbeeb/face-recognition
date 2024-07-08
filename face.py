import cv2
import face_recognition
import numpy as np

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces using Haar cascades
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Load known images and get face encodings
known_image = face_recognition.load_image_file("m1.jpeg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Load an image to recognize faces in
unknown_image = face_recognition.load_image_file("m2.jpeg")
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert to BGR for OpenCV
unknown_image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Face detection with Haar cascades
detected_faces = detect_faces(unknown_image_bgr)

# Draw rectangles around detected faces
for (x, y, w, h) in detected_faces:
    cv2.rectangle(unknown_image_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Face recognition
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
    name = "Unknown"

    if True in matches:
        name = "Known Person"

    # Draw a box around the face
    cv2.rectangle(unknown_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    # Draw a label with a name below the face
    cv2.rectangle(unknown_image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(unknown_image_bgr, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display the result
cv2.imshow('Face Detection and Recognition', unknown_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
