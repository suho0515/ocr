#!/usr/bin/env python3
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import imutils

class OCR:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

    model_dir = '/root/catkin_ws/src/ocr/model/knn_trained_model.xml'
    self.model = cv2.ml.KNearest_create()
    self.model = cv2.ml.KNearest_load(model_dir)

    self.old_result = [0, 0, 0]
    self.result = [-1, -1, -1]
    self.result_cnt = [[],[],[]]
    self.result_max = [-1, -1, -1]
    self.max_param = 5
    self.tracked_result = [0, 0, 0]

    self.start_cnt = 0
    

  def callback(self,data):
    self.result = [-1, -1, -1]

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      cv_image = cv2.flip(cv_image,-1)
      cv2.imshow("cv_image", cv_image)
    except CvBridgeError as e:
      print(e)


    try:
      height, width, channel = cv_image.shape
      matrix = cv2.getRotationMatrix2D((width/2, height/2), -3, 1)
      dst = cv2.warpAffine(cv_image, matrix, (width, height))
      # set roi
      x=260; y=90; w=50; h=180
      roi_img = dst[y:y+h, x:x+w]     
      cv2.imshow('roi_img', roi_img)

      # hsv convert
      hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

      # inrange for red number and black number
      rng_1 = cv2.inRange(hsv, (0, 50, 0), (255, 255, 200))
      rng_2 = cv2.inRange(hsv, (0, 0, 0), (255, 255, 100))

      cv2.imshow('rng_1', rng_1)
      cv2.imshow('rng_2', rng_2)

      # sum each ranged image
      sum = rng_1 + rng_2
      cv2.imshow('sum', sum)

      # kernel = np.ones((3, 3), np.uint8)
      # erosion = cv2.erode(sum, kernel, iterations=1)  #// make erosion image
      # cv2.imshow('erosion', erosion)

      # dilation
      #kernel = np.ones((3, 3), np.uint8)
      #dilation = cv2.dilate(sum, kernel, iterations=1)  #// make dilation image
      #cv2.imshow('dilation', dilation)

      # Removing noise
      # https://pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/

      cnts = cv2.findContours(sum,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      mask = np.ones(sum.shape[:2], dtype="uint8") * 255

      # loop over the contours
      for c in cnts:
        # if the contour is bad, draw it on the mask
        if self.is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)

      # remove the contours from the image and show the resulting images
      sum = cv2.bitwise_and(sum, sum, mask=mask)
      cv2.imshow("Mask", mask)
      cv2.imshow("After", sum)

      # https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
      thresh = cv2.adaptiveThreshold(sum,255,1,1,3,2)
      cv2.imshow('thresh', thresh)

      contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

      out = np.zeros(sum.shape,np.uint8)

      for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>10 and area<600:
            [x,y,w,h] = cv2.boundingRect(cnt)
            #M = cv2.moments(cnt)
            #cX = int(M["m10"] / M["m00"])
            #cY = int(M["m01"] / M["m00"])
            cX = x+w/2
            cY = y+h/2
            if  h>20 and h<50 and w<40 :
                cv2.rectangle(roi_img,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(40,40))
                roismall = roismall.reshape((1,1600))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = self.model.findNearest(roismall, k = 3)
                string = str(int((results[0][0])))
                if cX > 20 and cX < 30:
                  if (cY>15 and cY<45):
                    self.result[0] = int((results[0][0]))
                    if len(self.result_cnt[0]) < self.max_param:
                      self.result_cnt[0].append(int((results[0][0])))
                    else:
                      self.result_cnt[0].pop(0)
                      self.result_cnt[0].append(int((results[0][0])))
                      self.result_max[0] = max(self.result_cnt[0])
                      #print(self.result_max[0])
                    cv2.putText(out,string,(x,y+h),0,1,(255,255,255))
                  elif (cY>75 and cY<105):
                    self.result[1] = int((results[0][0]))
                    if len(self.result_cnt[1]) < self.max_param:
                      self.result_cnt[1].append(int((results[0][0])))
                    else:
                      self.result_cnt[1].pop(0)
                      self.result_cnt[1].append(int((results[0][0])))
                      self.result_max[1] = max(self.result_cnt[1])
                      #print(self.result_max[1])
                    cv2.putText(out,string,(x,y+h),0,1,(255,255,255))
                  elif (cY>135 and cY<165):
                    self.result[2] = int((results[0][0]))
                    if len(self.result_cnt[2]) < self.max_param:
                      self.result_cnt[2].append(int((results[0][0])))
                    else:
                      self.result_cnt[2].pop(0)
                      self.result_cnt[2].append(int((results[0][0])))
                      self.result_max[2] = max(self.result_cnt[2])
                      #print(self.result_max[2])
                    cv2.putText(out,string,(x,y+h),0,1,(255,255,255))
                
            #debug_str = 'area:' + str(area) + "/cX:" + str(cX) + "/cY" + str(cY) + "/h" + str(h)
            #print(debug_str)
            #print(self.result)

      cv2.imshow('final_image',roi_img)

      cv2.imshow('out',out)
      #cv2.waitKey(0)

      cv2.waitKey(3)

      # chase volume
      if self.start_cnt < 5:
        self.start_cnt = self.start_cnt + 1
        self.tracked_result = self.result
        self.old_result = self.tracked_result
      else:
        if (self.result_max[0] != -1) and (self.result_max[1] != -1) and (self.result_max[2] != -1):
          rospy.loginfo('1:%s/ 2:%s/ 3:%s' % (str(self.result_max[0]), str(self.result_max[1]), str(self.result_max[2])))
          
          
        
        #print(self.old_result)
        #print(self.result)
        #print("\n")
        # if ((self.result[0] < self.old_result[0]-1) \
        #   or (self.result[0] > self.old_result[0]+1)): sys.exit(1)
        # if ((self.result[1] < self.old_result[1]-1) \
        #   or (self.result[1] > self.old_result[1]+1)): sys.exit(1)
        # if ((self.result[2] < self.old_result[2]-1) \
        #   or (self.result[2] > self.old_result[2]+1)): sys.exit(1)
        # if ((self.result[0] >= self.tracked_result[0]-1) \
        #   and (self.result[0] <= self.tracked_result[0]+1)) \
        #   and ((self.result[1] >= self.tracked_result[1]-1) \
        #   and (self.result[1] <= self.tracked_result[1]+1)) \
        #   and ((self.result[2] >= self.tracked_result[2]-1) \
        #   and (self.result[2] <= self.tracked_result[2]+1)):
        #   self.tracked_result = self.result
        #   print(self.tracked_result)

        #print(self.result)
      
      self.old_result = self.result
    

    except CvBridgeError as e:
      print(e)

  def is_contour_bad(self, c):
	# approximate the contour
    if cv2.contourArea(c)>10:
        [x,y,w,h] = cv2.boundingRect(c)
        if  h>20 and h<40 and w<40:
            return False

	# the contour is 'bad' if it is not digit
    return True

def main(args):
  rospy.init_node('ocr_node', anonymous=True)
  ocr = OCR()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)