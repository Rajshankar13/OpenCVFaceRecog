import cv2
import numpy as np

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #classifier
cam = cv2.VideoCapture(0)

while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   #RGB to Gray

    faces = face.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (123, 0, 255), 2)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
    
