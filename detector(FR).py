import cv2
import numpy as np
import pickle

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #classifier
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\TrainingData.yml")

with open('details.pickle', 'rb') as f:
    details_dict = pickle.load(f)

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontColor = (255, 0 , 0)

while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   #RGB to Gray

    faces = face.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (123, 0, 255), 2)
        id, config = recognizer.predict(gray[y:y+h, x:x+w])

        cv2.putText(img, details_dict[id], (x, y+h), font, fontscale, fontColor)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()