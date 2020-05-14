import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.createLBPHFaceRecognizer();
rec.load("recognizer\\trainingData.yml")
id = 0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

while (True):
	ret, img = cam.read();
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0), 2 )
		id, conf = rec.predict(gray[y:y+h, x:x+w])
		if(id == 1):
			id = "EDDY"
		elif(id == 2):
		    id = "KISS EDDY"	
		cv2.putText(img, str(id), (x,y+h), fontface, fontscale, fontcolor) 
		
	cv2.imshow('Faces', img);  
	if(cv2.waitKey(1) == ord('q')):
		break; 

cam.release()
cv2.destroyAllWindows()