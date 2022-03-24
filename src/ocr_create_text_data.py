import cv2
import numpy as np

num_dir = 0
samples =  np.empty((0,100))
responses = []
while num_dir < 10:
    num = 0
    while num < 100:
        name = '/root/catkin_ws/src/ocr/dataset/'+ str(num_dir) + '/' + str(num) + '.jpg'
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('img',img)

        (thresh, bin_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow('bin_img',bin_img)

        #thresh = cv2.adaptiveThreshold(img,255,1,1,3,2)
        #rng = cv2.inRange(img, (0, 0, 0), (100, 100, 100))
        #cv2.imshow('rng',rng)
        #cv2.waitKey(10)
        
        contours,hierarchy = cv2.findContours(bin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt)>20:
                [x,y,w,h] = cv2.boundingRect(cnt)

                if  h>20:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    roi = bin_img[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    cv2.imshow('roismall',roismall)
                    cv2.waitKey(10)

        responses.append(int(num_dir))
        #print(int(num/10))
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)
        
        #cv2.imshow('img',img)
        #cv2.waitKey(0)

        num = num+1
    num_dir = num_dir+1

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('/root/catkin_ws/src/ocr/dataset/generalsamples.data',samples)
np.savetxt('/root/catkin_ws/src/ocr/dataset/generalresponses.data',responses)

cv2.destroyAllWindows()