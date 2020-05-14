import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

id = input('enter user id  ')
sampleNum = 0;
while (True):
	ret, img = cam.read();
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		sampleNum = sampleNum + 1
		cv2.imwrite("dataset/user."+ str(id) +"."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w] )
		cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0), 2 )
		cv2.waitKey(100)
	cv2.imshow("Faces", img)  
	cv2.waitKey(1)
	if (sampleNum == 4):
		break; 

cam.release()
cv2.destroyAllWindows()