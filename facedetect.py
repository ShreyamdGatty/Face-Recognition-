import cv2
import numpy as np

def examp():
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    SampleNum = 0
    id = input("enter user id")
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            SampleNum = SampleNum+1
            cv2.imwrite("Dataset/User."+str(id)+"."+str(SampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
            cv2.waitKey(1)
        
        cv2.imshow("Capture", img)
        cv2.waitKey(1)
        if(SampleNum>30):
            break
    cam.release()
    cv2.destroyAllWindows()

examp()
