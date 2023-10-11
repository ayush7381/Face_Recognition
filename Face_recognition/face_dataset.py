import cv2
import os

# Initialize the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Load the Haar Cascade classifier for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Prompt the user to enter a user id
face_id = input('\n Enter user id and press enter: ')

print("\n [INFO] Initializing face capture.")

# Initialize a count to keep track of captured images
count = 0

while True:
    # Read a frame from the camera
    ret, img = cam.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the detected face to the dataset folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        # Display the image with the detected face
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
