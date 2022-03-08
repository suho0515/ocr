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
    self.pre_vol_array = [-1, -1, -1]
    self.str_valid_vol = ""
    self.str_predicted_val_1 = ""
    self.str_predicted_val_2 = ""
    self.str_predicted_val_3 = ""
    self.str_rt_vol = ""
    self.init_vol_flag = False

    self.pub = rospy.Publisher('volume', Int8, queue_size=1)

    self.image_pub = rospy.Publisher("semantic_image",Image)
    

  def callback(self,data):
    self.result = [-1, -1, -1]

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      cv_image = cv2.flip(cv_image,-1)
      #cv2.imshow("cv_image", cv_image)
    except CvBridgeError as e:
      print(e)

    try:
      height, width, channel = cv_image.shape
      matrix = cv2.getRotationMatrix2D((width/2, height/2), 0, 1)
      dst = cv2.warpAffine(cv_image, matrix, (width, height))

      roi = self.get_roi(cv_image)
      h, w, c = roi.shape
      #print(h, w)

      #print(np.sum(roi[0:h, 0:w]))
      #cv2.imshow('roi', roi)
      if(np.sum(roi[0:h, 0:w]) != 0):
        semantic_image = roi.copy()
        black_img = np.zeros([semantic_image.shape[0],200,3],dtype=np.uint8)
        semantic_image = np.concatenate((semantic_image, black_img), axis=1)

        roi_1, roi_2, roi_3, pts_each = self.get_each_roi(roi)
        #cv2.imshow("roi_1", roi_1)
        #cv2.imshow("roi_2", roi_2)
        #cv2.imshow("roi_3", roi_3)
        
        vol = 0
        semantic_image, vol_array = self.detect_volume(roi_1, roi_2, roi_3, semantic_image, pts_each)
        print(vol_array)

        #print(str_rt_vol)

        if(vol_array[0] == -1):
          self.str_rt_vol = "X"
        else:
          self.str_rt_vol = str(vol_array[0])

        if(vol_array[1] == -1):
          self.str_rt_vol = self.str_rt_vol + "X"
        else:
          self.str_rt_vol = self.str_rt_vol + str(vol_array[1])

        if(vol_array[2] == -1):
          self.str_rt_vol = self.str_rt_vol + "X"
        else:
          self.str_rt_vol = self.str_rt_vol + str(vol_array[2])


        if((vol_array[0] != -1) and (vol_array[1] != -1) and (vol_array[2] != -1)):
          vol = vol_array[0]*100 + vol_array[1]*10 + vol_array[2]
        else:
          vol = None
        #print(vol_array)
        if(vol != None):

          tracked_vol = self.track_volume(vol)
          #print(tracked_vol)

          if(tracked_vol != None):
            
            #self.str_valid_vol = str_result
            self.str_valid_vol = str(self.pre_vol).zfill(3)
            self.str_predicted_val_1 = str(tracked_vol-1).zfill(3)
            self.str_predicted_val_2 = str(tracked_vol).zfill(3)
            self.str_predicted_val_3 = str(tracked_vol+1).zfill(3)

            self.pub.publish(tracked_vol)

          #cv2.waitKey(3)
        # Input Semantic Info
        str_valid_val = "valid value: " + self.str_valid_vol
        str_rt_val = "real-time value: " + self.str_rt_vol       
        str_predicted_val = "predicted value: "
        str_target_val = "target value: "

        cv2.putText(semantic_image,str_valid_val,(130, 50),0,0.5,(255,255,255),1)
        cv2.putText(semantic_image,str_rt_val,(130, 100),0,0.5,(255,255,255),1)
        cv2.putText(semantic_image,str_predicted_val,(130, 150),0,0.5,(255,255,255),1)
        cv2.putText(semantic_image,self.str_predicted_val_1,(265, 170),0,0.5,(255,255,255),1)
        cv2.putText(semantic_image,self.str_predicted_val_2,(265, 190),0,0.5,(255,255,255),1)
        cv2.putText(semantic_image,self.str_predicted_val_3,(265, 210),0,0.5,(255,255,255),1)
        cv2.putText(semantic_image,str_target_val,(130, 240),0,0.5,(255,255,255),1)
        # publish semantic image message
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(semantic_image, "bgr8"))

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
    h, w, c = img.shape
    result = np.zeros((w,h,1), dtype="uint8")
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
    
    if(len(cnts) == 4):
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
    #if(len(cnts) == 4) :
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
    pts_each = [[],[],[]]

    x=40; y=65; w=40; h=40
    roi_1 = roi[y:y+h, x:x+w]

    pts_each[0].append(x)
    pts_each[0].append(y)

    x=40; y=125; w=40; h=40
    roi_2 = roi[y:y+h, x:x+w]

    pts_each[1].append(x)
    pts_each[1].append(y)

    x=40; y=185; w=40; h=31
    roi_3 = roi[y:y+h, x:x+w]

    pts_each[2].append(x)
    pts_each[2].append(y)

    return roi_1, roi_2, roi_3, pts_each
    
  def detect_volume(self, roi_1, roi_2, roi_3, semantic_image, pts_each):
    error_flag = False
    vol_array = [-1, -1, -1]

    if((self.pre_vol_array[0] == -1) and (self.pre_vol_array[1] == -1) and (self.pre_vol_array[2] == -1)):
      result, semantic_image = self.detect_each_volume(roi_1, 100, semantic_image, pts_each[0])
      if(result == -1):
        error_flag = True
      else:
        # check num_1
        # it should be 0 or 1 only
        if(result != 1):
          result = 0
        vol_array[0] = result

      result, semantic_image = self.detect_each_volume(roi_2, 90, semantic_image, pts_each[1])
      #print(result)
      if(result == -1):
        error_flag = True
      else:
        vol_array[1] = result

      result, semantic_image = self.detect_each_volume(roi_3, 90, semantic_image, pts_each[2])
      #print(result)
      if(result == -1):
        error_flag = True
      else:
        vol_array[2] = result

      

    else:
      result, semantic_image = self.detect_each_volume(roi_1, 100, semantic_image, pts_each[0])
      if(result == -1):
        if(self.pre_vol_array[0] != -1):
          result = self.pre_vol_array[0]
          #print(self.pre_vol_array[0])
          vol_array[0] = result
        error_flag = True
      else:
        # check num_1
        # it should be 0 or 1 only
        if(result != 1):
          result = 0
        vol_array[0] = result

      result, semantic_image = self.detect_each_volume(roi_2, 90, semantic_image, pts_each[1])
      if(result == -1):
        error_flag = True
      else:
        # check num_2
        # it should be +1 or -1 from previous value
        if(self.pre_vol_array[1] == 9):
          if(result != 0 and result != 9 and result != 8):
            result = -1
            error_flag = True
        elif(self.pre_vol_array[1] ==0):
          if(result != 1 and result != 0 and result != 9):
            result = -1
            error_flag = True
        else:
          if((result < self.pre_vol_array[1]-1) or (result > self.pre_vol_array[1]+1)):
            result = -1
            error_flag = True
        vol_array[1] = result
          

      result, semantic_image = self.detect_each_volume(roi_3, 90, semantic_image, pts_each[2])
      if(result == -1):
        error_flag = True
      else:
        # check num_2
        # it should be +1 or -1 from previous value
        if(self.pre_vol_array[2] == 9):
          if(result != 0 and result != 9 and result != 8):
            result = -1
            error_flag = True
        elif(self.pre_vol_array[2] ==0):
          if(result != 1 and result != 0 and result != 9):
            result = -1
            error_flag = True
        else:
          if((result < self.pre_vol_array[2]-1) or (result > self.pre_vol_array[2]+1)):
            result = -1
            error_flag = True
        vol_array[2] = result

    
    if(error_flag):
      return semantic_image, vol_array
    else:
      self.pre_vol_array[0] = int(vol_array[0])
      self.pre_vol_array[1] = int(vol_array[1])
      self.pre_vol_array[2] = int(vol_array[2])
      #print(self.pre_vol_array)
      return semantic_image, vol_array

  def detect_each_volume(self, roi, thresh_value, semantic_image, pt_each):
    result_val = None
    roi_h, roi_w, roi_c = roi.shape

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #cv2.imshow("h", h)
    #cv2.imshow("s", s)
    #cv2.imshow("v", v)
    (thresh, bin_img) = cv2.threshold(v, thresh_value, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("bin", bin_img)

    contours,hierarchy = cv2.findContours(bin_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(roi.shape,np.uint8)
    draw_flag = False
    #print(len(contours))
    for cnt in contours:
      if cv2.contourArea(cnt)>20:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>20:
          # compare with center point
          M = cv2.moments(cnt, False)
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])

          in_center = self.judge_center(cx, cy, roi_w, roi_h)
          #print(cx)
          #print(in_center)

          if(in_center == False):
            return -1, semantic_image

          cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
          # semantic image drawing
          #print(pts)
          if(draw_flag == False):
            cv2.rectangle(semantic_image,(int(pt_each[0])+x,int(pt_each[1])+y),(int(pt_each[0])+x+w,int(pt_each[1])+y+h),(0,255,0),2)
            draw_flag = True
          roi_bin = bin_img[y:y+h,x:x+w]
          #cv2.imshow('roi_bin',roi_bin)
          roismall = cv2.resize(roi_bin,(10,10))
          #cv2.imshow('roismall',roismall)
          roismall = roismall.reshape((1,100))
          roismall = np.float32(roismall)
          
          retval, results, neigh_resp, dists = self.model.findNearest(roismall, k = 3)
          string = str(int((results[0][0])))
          #print(string)
          #cv2.putText(out,string,(x,y+h),0,1,(255,255,255))

          result_val = int((results[0][0]))
          #cv2.imshow("roi",roi)
          #cv2.waitKey(3)

    if(result_val == None):
      return -1, semantic_image
    else:
      return result_val, semantic_image

  def judge_center(self, cx, cy, w, h):
    if((cx > w/4) and (cx < w*(3/4))):
      return True
    else: 
      return False

  def track_volume(self, vol):
    if (len(self.result_cnt) < self.max_param) and (self.init_vol_flag == False):
      self.result_cnt.append(vol)
    else :
      if(self.init_vol_flag == False):
        #self.result_cnt.pop(0)
        #self.result_cnt.append(vol)
        self.result_max = max(self.result_cnt)
        #print(self.result_max)
        self.pre_vol = self.result_max

      self.init_vol_flag = True

    if(self.init_vol_flag == True):
      if((vol == self.pre_vol) or (vol == self.pre_vol-1) or (vol == self.pre_vol+1)):
        self.pre_vol = vol
        #print(self.pre_vol)
        return vol
      else: 
        return self.pre_vol
        


      

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



