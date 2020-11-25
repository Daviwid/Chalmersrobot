import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_RGBA2RGB)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(50) == ord('x'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()