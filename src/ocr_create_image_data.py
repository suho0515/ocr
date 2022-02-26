import numpy as np
import imutils
import cv2

def is_contour_bad(c):
	# approximate the contour
    if cv2.contourArea(c)>50:
        [x,y,w,h] = cv2.boundingRect(c)
        if  h>20:
            return False

	# the contour is 'bad' if it is not digit
    return True

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

number = 0
while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    #cv2.imshow("VideoFrame", frame)

    cv_image = cv2.flip(frame,-1)
    #cv2.imshow("cv_image", cv_image)

    x=300; y=240; w=40; h=40
    roi_img = cv_image[y:y+h, x:x+w]     
    cv2.imshow('roi_img', roi_img)

    # 
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    rng_1 = cv2.inRange(hsv, (0, 100, 0), (255, 255, 150))
    rng_2 = cv2.inRange(hsv, (0, 0, 0), (255, 255, 80))

    cv2.imshow('rng_1', rng_1)
    cv2.imshow('rng_2', rng_2)

    sum = rng_1 + rng_2
    cv2.imshow('sum', sum)

    # Removing noise
    # https://pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/

    cnts = cv2.findContours(sum,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.ones(sum.shape[:2], dtype="uint8") * 255

    # loop over the contours
    for c in cnts:
        # if the contour is bad, draw it on the mask
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)

    # remove the contours from the image and show the resulting images
    sum = cv2.bitwise_and(sum, sum, mask=mask)
    cv2.imshow("Mask", mask)
    cv2.imshow("After", sum)
    # https://hoony-gunputer.tistory.com/entry/OpenCv-python-%EA%B8%B0%EC%B4%88-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9D%BD%EA%B3%A0-%EC%A0%80%EC%9E%A5%ED%95%98%EA%B8%B0
    name = '/root/catkin_ws/src/ocr/dataset/' + str(number) + '.jpg'
    cv2.imwrite(name, sum)
    cv2.waitKey(0)
    
    number = number + 1


capture.release()
cv2.destroyAllWindows()