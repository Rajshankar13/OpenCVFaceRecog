import cv2
import numpy as np
import pickle

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #classifier
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\TrainingData.yml")

details_data = []
with open('details.pickle', 'rb') as fr:
    try:
        while True:
            details_data.append(pickle.load(fr))
    except EOFError:
        pass

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontColor = (255, 0 , 0)

def find_name_by_id(ix):
    for d in details_data:
        if(ix in d.keys()):
            name = d[ix]
    return name


while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   #RGB to Gray

    faces = face.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (123, 0, 255), 2)
        id, config = recognizer.predict(gray[y:y+h, x:x+w])
        #print(id, config)      To set the confidence, see the output of config and modify it accordingly
        if(config < 100):            
            name = find_name_by_id(id)
        else:
            name = 'No match'

        cv2.putText(img, name, (x, y+h), font, fontscale, fontColor)
    cv2.imshow("Face", img)
    
    if(cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()