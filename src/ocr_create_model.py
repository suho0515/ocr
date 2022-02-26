import cv2
import numpy as np

cnt = 0

samples =  np.empty((0,100))
responses = []

while cnt < 100:
    name = '/root/catkin_ws/src/ocr/dataset/' + str(cnt) + '.jpg'
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    roismall = cv2.resize(img,(10,10))

    cv2.imshow('roismall',roismall)

    responses.append(int(cnt/10))
    #print(int(cnt/10))
    sample = roismall.reshape((1,100))
    samples = np.append(samples,sample,0)
    
    #cv2.imshow('img',img)
    #cv2.waitKey(0)

    cnt = cnt+1

    if cnt==100:
        responses = np.array(responses,np.float32)
        responses = responses.reshape((responses.size,1))
        print("training complete")

        np.savetxt('/root/catkin_ws/src/ocr/model/generalsamples.data',samples)
        np.savetxt('/root/catkin_ws/src/ocr/model/generalresponses.data',responses)
    

cv2.destroyAllWindows()