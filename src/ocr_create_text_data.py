import cv2
import numpy as np

num = 0

samples =  np.empty((0,100))
responses = []

while num < 100:
    name = '/root/catkin_ws/src/ocr/dataset/' + str(num) + '.jpg'
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt)>100:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if  h>20:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                roi = img[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('roismall',roismall)

    responses.append(int(num/10))
    #print(int(num/10))
    sample = roismall.reshape((1,100))
    samples = np.append(samples,sample,0)
    
    #cv2.imshow('img',img)
    #cv2.waitKey(0)

    num = num+1

    if num==100:
        responses = np.array(responses,np.float32)
        responses = responses.reshape((responses.size,1))
        print("training complete")

        np.savetxt('/root/catkin_ws/src/ocr/dataset/generalsamples.data',samples)
        np.savetxt('/root/catkin_ws/src/ocr/dataset/generalresponses.data',responses)
    

cv2.destroyAllWindows()