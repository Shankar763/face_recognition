#pylint:disable=no-member
import numpy as np
import cv2 as cv
import datetime
import os
haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Shankar Yadav', 'Beruit Ban', 'Prabin Pokhrel']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
cap=cv.VideoCapture(0)
# n=0;
while(cap.isOpened()):
    ret,img=cap.read();
    if ret==True:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('Person', gray)
        # cv.imwrite(r'access_image\1.jpg',img)
    # if n==1:
        # break
    # n=n+1
        

# cap.release()        
# img = cv.imread(pyp

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect the face in the image
        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h,x:x+w]

            label, confidence = face_recognizer.predict(faces_roi)
            print(f'Label = {people[label]} with a confidence of {confidence}')

            cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            dated=str(datetime.datetime.now())
            font=cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img,dated,(60,50),font,1,(0,0,0),2,cv.LINE_AA)
            cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

            cv.imshow('Detected Face', img)

    if cv.waitKey(1) & 0xFF == ord('e'):
        break 

cv.destroyAllWindows()
cap.release()
         