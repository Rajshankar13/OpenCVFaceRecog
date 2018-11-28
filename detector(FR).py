import cv2
import numpy as np

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #classifier
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\TrainingData.yml")
id = 0 
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
        if(id == 13):
            id = "Rajshankar!"
        elif(id == 2):
            id = "Barack Obama!"
        elif(id == 7):
            id = "Ryan!"
        elif(id == 18):
            id = "Suman"
        elif(id == 123):
            id = "Nimmmy"
        elif(id == 133):
            id = "Life!"
        cv2.putText(img, str(id), (x, y+h), font, fontscale, fontColor)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
    
