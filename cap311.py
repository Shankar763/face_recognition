import cv2 as cv
import os
cap=cv.VideoCapture(0,cv.CAP_DSHOW)
num=0

while(cap.isOpened()):
    
    # if ret==True:
    
    # cv.SaveImage(r'C:\Users\user\faces\unknown'+str(num)+'.jpg', frame)
    if not os.path.exists('faces/Unknown'+str(num)):
        print('Folder create')
        os.mkdir('faces/Unknown'+str(num))
        # n=0
        for i in range(101):
            ret,frame=cap.read()
            # if ret==True:
            cv.imshow('frame',frame)
            print('images saving..')
            

            savepath='faces/Unknown'+str(num)

            filepath1=os.path.join(savepath,str(i)+'.jpg')
            cv.imwrite(filepath1,frame)
            # n+=1
        cap.release()
        cv.destroyAllWindows()
        break
    num += 1