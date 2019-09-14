import cv2
import sys
# cascPath = sys.argv[1]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

videoCapture = cv2.VideoCapture(0)

videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 300);
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300);

while True:
    # Capture frame-by-frame
    ret, frame = videoCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.03, 3)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
videoCapture.release()
cv2.destroyAllWindows()
