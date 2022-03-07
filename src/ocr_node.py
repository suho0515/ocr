#!/usr/bin/env python3
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Int8
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

    self.result_cnt = []
    self.max_param = 10
    self.result_max = -1
    self.pre_vol = -1
    self.init_vol_flag = False

    self.pub = rospy.Publisher('volume', Int8, queue_size=10)


    

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
      
      roi = self.get_roi(cv_image)
      cv2.imshow('roi', roi)

      roi_1, roi_2, roi_3 = self.get_each_roi(roi)
      cv2.imshow("roi_1", roi_1)
      cv2.imshow("roi_2", roi_2)
      cv2.imshow("roi_3", roi_3)

      vol = self.detect_volume(roi_1, roi_2, roi_3)

      tracked_vol = self.track_volume(vol)
      print(tracked_vol)

      self.pub.publish(tracked_vol)

      cv2.waitKey(100)

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

  def get_roi(self, img):

    marked_img = img.copy()

    # inrange for red number and black number
    rng = cv2.inRange(img, (100, 100, 100), (255, 255, 255))

    # Removing noise
    # https://pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/

    cnts = cv2.findContours(rng,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.ones(rng.shape[:2], dtype="uint8") * 255

    # loop over the contours
    for c in cnts:
      # if the contour is bad, draw it on the mask
      if self.is_contour_bad(c):
          cv2.drawContours(mask, [c], -1, 0, -1)

    # remove the contours from the image and show the resulting images
    rng = cv2.bitwise_and(rng, rng, mask=mask)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("After", rng)


    # find 4 contours
    pts_cnt = 0
    pts = np.zeros((4, 2), dtype=np.float32)

    cnts, hierarchy = cv2.findContours(rng,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for i in cnts:
      M = cv2.moments(i)
      cX = int(M['m10'] / M['m00'])
      cY = int(M['m01'] / M['m00'])
      
      cv2.circle(marked_img, (cX, cY), 3, (255, 0, 0), -1)
      cv2.drawContours(marked_img, [i], 0, (0, 0, 255), 2)

      pts[pts_cnt] = [cX, cY]
      pts_cnt += 1
    
    #cv2.imshow("marked_img", marked_img)

    # perspective transform
    # https://minimin2.tistory.com/135
    #print(len(cnts))
    if(len(cnts) == 4) :
      sm = pts.sum(axis=1)
      #print(sm)
      diff = np.diff(pts, axis=1)

      topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
      bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
      topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
      bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

      # 변환 전 4개 좌표 
      pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

      # 변환 후 영상에 사용할 서류의 폭과 높이 계산
      w1 = abs(bottomRight[0] - bottomLeft[0])
      w2 = abs(topRight[0] - topLeft[0])
      h1 = abs(topRight[1] - bottomRight[1])
      h2 = abs(topLeft[1] - bottomLeft[1])
      width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
      height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

      # 변환 후 4개 좌표
      pts2 = np.float32([[0, 0], [width - 1, 0],
                          [width - 1, height - 1], [0, height - 1]])

      # 변환 행렬 계산 
      mtrx = cv2.getPerspectiveTransform(pts1, pts2)
      # 원근 변환 적용
      result = cv2.warpPerspective(img, mtrx, (width, height))
      #cv2.imshow('scanned', result)

    return result
  
  def get_each_roi(self, roi):
    x=40; y=65; w=40; h=40
    roi_1 = roi[y:y+h, x:x+w]

    x=40; y=125; w=40; h=40
    roi_2 = roi[y:y+h, x:x+w]

    x=40; y=185; w=40; h=37
    roi_3 = roi[y:y+h, x:x+w]

    return roi_1, roi_2, roi_3
    
  def detect_volume(self, roi_1, roi_2, roi_3):
    result_val = 0

    hsv = cv2.cvtColor(roi_1, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #cv2.imshow("h", h)
    #cv2.imshow("s", s)
    #cv2.imshow("v", v)
    (thresh, bin_img) = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("bin", bin_img)

    contours,hierarchy = cv2.findContours(bin_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(roi_1.shape,np.uint8)
    for cnt in contours:
      if cv2.contourArea(cnt)>20:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>20:
          cv2.rectangle(roi_1,(x,y),(x+w,y+h),(0,255,0),2)
          roi = bin_img[y:y+h,x:x+w]
          #cv2.imshow('roi',roi)
          roismall = cv2.resize(roi,(10,10))
          cv2.imshow('roismall',roismall)
          roismall = roismall.reshape((1,100))
          roismall = np.float32(roismall)
          
          retval, results, neigh_resp, dists = self.model.findNearest(roismall, k = 3)
          string = str(int((results[0][0])))
          #print(string)
          #cv2.putText(out,string,(x,y+h),0,1,(255,255,255))

          result_val = int((results[0][0]))*100

            
    hsv = cv2.cvtColor(roi_2, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #cv2.imshow("b", h)
    #cv2.imshow("g", s)
    #cv2.imshow("r", v)
    (thresh, bin_img) = cv2.threshold(v, 90, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("bin", bin_img)

    contours,hierarchy = cv2.findContours(bin_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(roi_2.shape,np.uint8)
    for cnt in contours:
      if cv2.contourArea(cnt)>20:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>20:
          cv2.rectangle(roi_2,(x,y),(x+w,y+h),(0,255,0),2)
          roi = bin_img[y:y+h,x:x+w]
          #cv2.imshow('roi',roi)
          roismall = cv2.resize(roi,(10,10))
          #cv2.imshow('roismall',roismall)
          roismall = roismall.reshape((1,100))
          roismall = np.float32(roismall)
          
          retval, results, neigh_resp, dists = self.model.findNearest(roismall, k = 3)
          string = str(int((results[0][0])))
          #print(string)
          cv2.putText(out,string,(x,y+h),0,1,(255,255,255))

          result_val = result_val + int((results[0][0]))*10

    hsv = cv2.cvtColor(roi_3, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #cv2.imshow("b", h)
    #cv2.imshow("g", s)
    #cv2.imshow("r", v)
    (thresh, bin_img) = cv2.threshold(v, 90, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("bin", bin_img)

    contours,hierarchy = cv2.findContours(bin_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(roi_3.shape,np.uint8)
    for cnt in contours:
      if cv2.contourArea(cnt)>20:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>20:
          cv2.rectangle(roi_3,(x,y),(x+w,y+h),(0,255,0),2)
          roi = bin_img[y:y+h,x:x+w]
          #cv2.imshow('roi',roi)
          roismall = cv2.resize(roi,(10,10))
          cv2.imshow('roismall',roismall)
          roismall = roismall.reshape((1,100))
          roismall = np.float32(roismall)
          
          retval, results, neigh_resp, dists = self.model.findNearest(roismall, k = 3)
          string = str(int((results[0][0])))
          #print(string)
          #cv2.putText(out,string,(x,y+h),0,1,(255,255,255)

          result_val = result_val + int((results[0][0]))
          
          #print(result_val)
          #cv2.waitKey()
    return result_val

  def track_volume(self, vol):
    if (len(self.result_cnt) < self.max_param) and (self.init_vol_flag == False):
      self.result_cnt.append(vol)
    else :
      if(self.init_vol_flag == False):
        #self.result_cnt.pop(0)
        #self.result_cnt.append(vol)
        self.result_max = max(self.result_cnt)
        print(self.result_max)
        self.pre_vol = self.result_max

      self.init_vol_flag = True

    if(self.init_vol_flag == True):
      if((vol == self.pre_vol) or (vol == self.pre_vol-1) or (vol == self.pre_vol+1)):
        self.pre_vol = vol
        #print(self.pre_vol)
        return vol
        


      

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



