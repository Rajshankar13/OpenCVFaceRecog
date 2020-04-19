import os
import cv2
import sys
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImage(path):
	ImgPaths = [os.path.join(path, f) for f in os.listdir(path)]
	faces = []
	ids = []
	for ImgPath in ImgPaths:
		faceImg = Image.open(ImgPath).convert('L')
		faceNp = np.array(faceImg, 'uint8')
		ID = int(os.path.split(ImgPath) [-1].split('.')[1])
		faces.append(faceNp)
		ids.append(ID)
		cv2.imshow('Training', faceNp)
		cv2.waitKey(10)
	return ids, faces

Ids, faces = getImage(path)
print(Ids)
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer/TrainingData.yml')
cv2.destroyAllWindows()