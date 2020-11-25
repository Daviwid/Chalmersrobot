import os
from cv2 import cv2
import numpy as np

#Use document for recognizing faces
face_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")

#Use camera to capture video
cap = cv2.VideoCapture(0)


#Countouring images function
def Convert(sketch):
    #Read the image which will be in gray already
    img = cv2.imread('portrait.png')
    #Invert the grayscaled image
    inv_gray_image = 255 - img
    #Blurr the image
    Blurred_image = cv2.GaussianBlur(inv_gray_image,(21,21),0)
    #Invert the blurred image
    inv_blur = 255 - Blurred_image
    #Divide the original gray image with the inverted blurred image and put in a variable
    sketch = cv2.divide(img, inv_blur, scale=256)
    #Return the variable
    return sketch

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Grayscale the frames which will be used in the function
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Find coordinates for the faces in the picture
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    #for-loop which will use the coordinates found in the image to draw out a frame around the faces found
    for (x, y, w, h) in faces:
        #Print out the coordinates
        print(x,y,w,h)
        #Region of intrest = coordinates around the face
        roi_gray = gray[(y-60):(y)+(h+40), (x-20):x+(w+20)]
        
        #Rectangle function
        #Color = blue
        color = (255, 0, 0)
        #Thickness
        stroke = 2
        #X + W line = from X coordinate plus the pixels to the W 
        end_cord_x = x + w
        #Y + H line = from Y coordinate plus the pixels to the H 
        end_cord_y = (y) + (h + 40)
        #Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y-60), (end_cord_x, end_cord_y), color, stroke)

    #Breaking the loop
    cv2.imshow('frame', frame)
    #If "q" pressed break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Save a picture as png
cv2.imwrite("portrait.png", roi_gray)

#Send to function
sketch = Convert("portrait.png")

#Overwrite the previous saved picture
cv2.imwrite("portrait.png", sketch)

#Stop recording
cap.release()
#Close down opened windows
cv2.destroyAllWindows()