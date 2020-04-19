import cv2
import numpy as np
import pickle
import time

details_dict = dict()
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #classifier
cam = cv2.VideoCapture(0)

id = int(time.time())
name = input('Enter Name: ')
details_dict[id] = name

sampleNum = 0

while(True):
	ret,img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   #RGB to Gray

	faces = face.detectMultiScale(gray, 1.3, 5)
	for x,y,w,h in faces:
		sampleNum+=1
		cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
		cv2.rectangle(img, (x, y), (x+w, y+h), (123, 0, 255), 2)
		cv2.waitKey(100)
	cv2.imshow("Face", img)
	cv2.waitKey(1)
	if(sampleNum > 20):
		break

cam.release()
cv2.destroyAllWindows()

with open('details.pickle', 'wb') as f:
	pickle.dump(details_dict, f)