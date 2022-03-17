#!/usr/bin/env python3

import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Int8
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import imutils

class OCR:

  def __init__(self):
    # Subscriber
    ## bridge which exist for converting ros image message data to opencv image format
    self.bridge = CvBridge()
    ## image subscriber which retreive topic message "/usb_cam/image_raw"
    self.img_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

    # Publisher
    ## raw image publisher. the topic message is "image_raw" which is not same with "/usb_cam/image_raw".(cause flip method is adapted) 
    self.img_pub = rospy.Publisher("image_raw",Image, queue_size=1)
    ## draft roi image publisher. the topic message is "draft_roi_image"
    self.draft_roi_img_pub = rospy.Publisher("draft_roi_image",Image, queue_size=1)
    ## big hsv image(in hsv format) publisher. the topic message is "big_hsv_image".
    self.big_hsv_img_pub = rospy.Publisher("big_hsv_img",Image, queue_size=1)
    ## big h image(in hsv format) publisher. the topic message is "big_h_image".
    self.big_h_img_pub = rospy.Publisher("big_h_img",Image, queue_size=1)
    ## big s image(in hsv format) publisher. the topic message is "big_s_image".
    self.big_s_img_pub = rospy.Publisher("big_s_img",Image, queue_size=1)
    ## big v image(in hsv format) publisher. the topic message is "big_v_image".
    self.big_v_img_pub = rospy.Publisher("big_v_img",Image, queue_size=1)
    ## big binary image publisher. the topic message is "big_binary_image".
    self.big_bin_img_pub = rospy.Publisher("big_binary_img",Image, queue_size=1)
    ## big noise removed binary image publisher. the topic message is "big_noise_removed_binary_image".
    self.big_noise_removed_bin_img_pub = rospy.Publisher("big_noise_removed_binary_image",Image, queue_size=1)
    ## big mask binary image publisher. the topic message is "big_mask_binary_image".
    self.big_mask_bin_img_pub = rospy.Publisher("big_mask_binary_image",Image, queue_size=1)



    # Variables
    ## self.debug_flag: if debug_flag is true, all debug topic message would published. if debug_flag is false, only necessary topic message would published.
    self.debug_flag = True


    self.state_sub = rospy.Subscriber("controller_state",Int8,self.stateCallback)



    model_dir = '/root/catkin_ws/src/ocr/model/knn_trained_model.xml'
    self.model = cv2.ml.KNearest_create()
    self.model = cv2.ml.KNearest_load(model_dir)

    self.result_cnt = []
    self.max_param = 3
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
    self.ctr_cmd_sub = rospy.Subscriber("command", Int16MultiArray, self.ctrCmdCallback)

    
    self.semantic_image_pub = rospy.Publisher("semantic_image",Image, queue_size=1)

    self.state = -1
    self.goal_vol = -1
    
  def ctrCmdCallback(self,cmd):
    self.goal_vol = cmd.data[5]
    print(cmd.data[5])
    print(self.goal_vol)
    

  def stateCallback(self,data):
    self.state = data.data
    #print(self.state)

  def callback(self,data):
    self.result = [-1, -1, -1]
    try:
      self.estimate_volume(data)

    except CvBridgeError as e:
      print(e)

    # try:
    #   height, width, channel = cv_image.shape
    #   matrix = cv2.getRotationMatrix2D((width/2, height/2), 0, 1)
    #   dst = cv2.warpAffine(cv_image, matrix, (width, height))

    #   roi = self.get_roi(cv_image)
    #   h, w, c = roi.shape
    #   #print(h, w)

    #   #print(np.sum(roi[0:h, 0:w]))
    #   #cv2.imshow('roi', roi)
    #   cv2.waitKey(10)
    #   if(np.sum(roi[0:h, 0:w]) != 0):
    #     semantic_image = roi.copy()
    #     black_img = np.zeros([semantic_image.shape[0],200,3],dtype=np.uint8)
    #     semantic_image = np.concatenate((semantic_image, black_img), axis=1)

    #     roi_1, roi_2, roi_3, pts_each = self.get_each_roi(roi)
    #     #cv2.imshow("roi_1", roi_1)
    #     #cv2.imshow("roi_2", roi_2)
    #     #cv2.imshow("roi_3", roi_3)
        
    #     vol = 0
    #     semantic_image, vol_array = self.detect_volume(roi_1, roi_2, roi_3, semantic_image, pts_each)
    #     print(vol_array)

    #     #print(str_rt_vol)

    #     if(vol_array[0] == -1):
    #       self.str_rt_vol = "X"
    #     else:
    #       self.str_rt_vol = str(vol_array[0])

    #     if(vol_array[1] == -1):
    #       self.str_rt_vol = self.str_rt_vol + "X"
    #     else:
    #       self.str_rt_vol = self.str_rt_vol + str(vol_array[1])

    #     if(vol_array[2] == -1):
    #       self.str_rt_vol = self.str_rt_vol + "X"
    #     else:
    #       self.str_rt_vol = self.str_rt_vol + str(vol_array[2])


    #     if((vol_array[0] != -1) and (vol_array[1] != -1) and (vol_array[2] != -1)):
    #       vol = vol_array[0]*100 + vol_array[1]*10 + vol_array[2]
    #     else:
    #       vol = None
    #     #print(vol_array)
    #     if(vol != None):

    #       tracked_vol = self.track_volume(vol)
    #       #print(tracked_vol)

    #       if(tracked_vol != None):
    #         # reset
    #         if(abs(tracked_vol - vol) > 3):
    #           self.result_cnt = []
    #           self.init_vol_flag = False
    #           self.pre_vol_array = [-1, -1, -1]
            
    #         #self.str_valid_vol = str_result
    #         self.str_valid_vol = str(self.pre_vol).zfill(3)
    #         self.str_predicted_val_1 = str(tracked_vol-1).zfill(3)
    #         self.str_predicted_val_2 = str(tracked_vol).zfill(3)
    #         self.str_predicted_val_3 = str(tracked_vol+1).zfill(3)

    #         self.pub.publish(tracked_vol)

    #       #cv2.waitKey(3)
    #     # Input Semantic Info
    #     str_valid_val = "valid value: " + self.str_valid_vol
    #     str_rt_val = "real-time value: " + self.str_rt_vol       
    #     str_predicted_val = "predicted value: "
    #     str_target_val = "target value: " + str(self.goal_vol)
    #     str_state = ""
    #     if(self.state == 1):
    #       str_state = "MOVE UP"
    #     elif(self.state == 2):
    #       str_state = "MOVE DOWN"
    #     elif(self.state == 3):
    #       str_state = "STOP"

    #     str_state_full = "state: " + str_state

    #     cv2.putText(semantic_image,str_valid_val,(130, 20),0,0.5,(255,255,255),1)
    #     cv2.putText(semantic_image,str_rt_val,(130, 70),0,0.5,(255,255,255),1)
    #     cv2.putText(semantic_image,str_predicted_val,(130, 120),0,0.5,(255,255,255),1)
    #     cv2.putText(semantic_image,self.str_predicted_val_1,(265, 140),0,0.5,(255,255,255),1)
    #     cv2.putText(semantic_image,self.str_predicted_val_2,(265, 160),0,0.5,(255,255,255),1)
    #     cv2.putText(semantic_image,self.str_predicted_val_3,(265, 180),0,0.5,(255,255,255),1)
    #     cv2.putText(semantic_image,str_target_val,(130, 210),0,0.5,(255,255,255),1)
    #     cv2.putText(semantic_image,str_state_full,(130, 260),0,0.5,(255,255,255),1)

    #     # publish semantic image message
    #     self.semantic_image_pub.publish(self.bridge.cv2_to_imgmsg(semantic_image, "bgr8"))

    # except CvBridgeError as e:
    #   print(e)


  def estimate_volume(self, data):
    cv_image = self.convert_to_cv_image(data)
    if(cv_image is None): return

    draft_roi_img = self.get_draft_roi(cv_image)
    if(draft_roi_img is None): return

    big_hsv_img, big_h_img, big_s_img, big_v_img = self.get_hsv(draft_roi_img)
    if((big_h_img is None) or (big_s_img is None) or (big_v_img is None)): return

    big_bin_img = self.get_binary(big_h_img)
    if(big_bin_img is None): return

    big_noise_removed_bin_img, big_mask_bin_img = self.remove_noise(big_bin_img)

    if(self.debug_flag):
      self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      self.draft_roi_img_pub.publish(self.bridge.cv2_to_imgmsg(draft_roi_img, "bgr8"))
      self.big_hsv_img_pub.publish(self.bridge.cv2_to_imgmsg(big_hsv_img, "bgr8"))
      self.big_h_img_pub.publish(self.bridge.cv2_to_imgmsg(big_h_img, "mono8"))
      self.big_s_img_pub.publish(self.bridge.cv2_to_imgmsg(big_s_img, "mono8"))
      self.big_v_img_pub.publish(self.bridge.cv2_to_imgmsg(big_v_img, "mono8"))
      self.big_bin_img_pub.publish(self.bridge.cv2_to_imgmsg(big_bin_img, "mono8"))
      self.big_noise_removed_bin_img_pub.publish(self.bridge.cv2_to_imgmsg(big_noise_removed_bin_img, "mono8"))
      self.big_mask_bin_img_pub.publish(self.bridge.cv2_to_imgmsg(big_mask_bin_img, "mono8"))
    pass

  def convert_to_cv_image(self, data):
    """!
    @brief when image callback function get the ros image message, convert it to cv_image(for opencv format)
    @details 

    @param[in] data: ros image message
    
    @note input constraints: 
    @n  - none

    @note output constraints: 
    @n                        - cv_image should not be empty 
    @n                        - cv_image should be sized 640 x 480 
    @n                        - cv_image should be color image (3 channel)

    @return cv_image converted to opencv format image
    """
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      if cv_image is None:
        raise NoImageError

      h, w, c = cv_image.shape
      if((h != 480) or (w != 640)):
        raise ImageSizeError(w, h)

      if (c != 3):
        raise ColorImageError(c)

      cv_image = cv2.flip(cv_image,-1)
      return cv_image
    except Exception as e:
      print(e)
      return None

  def get_draft_roi(self, img):
    """!
    @brief to avoid complexity of image processing, we get only draft roi.
    @details 

    @param[in] img: opencv format image
    
    @note input constraints: 
    @n - image should not be empty 
    @n - image should be sized 640 x 480 
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - draft_roi_image should not be empty 
    @n - draft_roi_image should be color image (3 channel)

    @return draft_roi_img: roi image which is given size manually, this approach sould be modificated in future
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      h, w, c = img.shape
      if((h != 480) or (w != 640)):
        raise ImageSizeError(w, h)
      if (c != 3):
        raise ColorImageError(c)

      # doing
      draft_roi_img = img[int(h*(1/20)):int(h*(16/20)), int(w*(6/20)):int(w*(13/20))]

      # output error check
      if draft_roi_img is None:
        raise NoImageError
      h, w, c = draft_roi_img.shape
      if (c != 3):
        raise ColorImageError(c)

      # return
      return draft_roi_img

    except Exception as e:
      print(e)
      return None
    pass

  def get_hsv(self, img):
    """!
    @brief for detecting marker, we pre-process image to hsv.
    @details the function return h, s, v image. for detecting marker, we're gonna use h image cause color of marker is yellow and it's opposite of micropipette body color(body color is indigo)

    @param[in] img: draft_roi_img should be input
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - h_image should not be empty 
    @n - h_image should be mono image (1 channel)
    @n - s_image should not be empty 
    @n - s_image should be mono image (1 channel)
    @n - v_image should not be empty 
    @n - v_image should be mono image (1 channel)

    @return h_img, s_img, v_img: all images sould be return
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      h, w, c = img.shape
      if (c != 3):
        raise ColorImageError(c)

      # doing
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      h_img, s_img, v_img = cv2.split(hsv)
      
      # output error check
      if (h_img is None) or (s_img is None) or (v_img is None):
        raise NoImageError
      if (h_img.dtype != 'uint8') or (s_img.dtype != 'uint8') or (v_img.dtype != 'uint8'):
        raise MonoImageError(h_img.dtype)

      # return
      return hsv, h_img, s_img, v_img

    except Exception as e:
      print(e)
      return None, None, None
    pass

  def get_binary(self, img):
    """!
    @brief before getting contour from image, we should convert gray image to binary image
    @details it is important to use apropriate image format(h or s or v), and set apropriate threshold value.

    @param[in] img: h or s or v image
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be gray image (1 channel)

    @note output constraints: 
    @n - binary_image should not be empty 
    @n - binary_image should be binary image (1 channel)

    @return bin_img: output binary image
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.dtype != 'uint8'):
        raise MonoImageError(c)

      # doing
      blur = cv2.GaussianBlur(img,(11,11),0)
      ret, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

      # output error check
      if (bin_img is None):
        raise NoImageError
      if (bin_img.dtype != 'uint8'):
        raise MonoImageError(bin_img.dtype)

      # return
      return bin_img

    except Exception as e:
      print(e)
      return None
    pass

  def remove_noise(self, img):
    """!
    @brief remove unnessary contour, so we prevent error and focus on important data
    @details this method reffered from "https://pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/"

    @param[in] img: binary image
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be binary image (1 channel)

    @note output constraints: 
    @n - noise_removed_binary_image should not be empty 
    @n - noise_removed_binary_image should be binary image (1 channel)

    @return noise_removed_bin_img: output binary image which the noise is removed
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.dtype != 'uint8'):
        raise MonoImageError(c)

      # doing
      # Removing noise
      # https://pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/

      cnts = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      mask = np.ones(img.shape[:2], dtype="uint8") * 255

      # loop over the contours
      for c in cnts:
        [x,y,w,h] = cv2.boundingRect(c)
        # if the contour is bad, draw it on the mask
        if cv2.contourArea(c)<1000 or cv2.contourArea(c)>2000 \
          or h<20 or h>60 or w<20 or w>60:
          cv2.drawContours(mask, [c], -1, 0, -1)
        
      # remove the contours from the image and show the resulting images
      noise_removed_bin_img = cv2.bitwise_and(img, img, mask=mask)
      #cv2.imshow("Mask", mask)
      #cv2.imshow("After", rng)

      # output error check
      if (noise_removed_bin_img is None):
        raise NoImageError
      if (noise_removed_bin_img.dtype != 'uint8'):
        raise MonoImageError(noise_removed_bin_img.dtype)

      # return
      return noise_removed_bin_img, mask

    except Exception as e:
      print(e)
      return None
    pass


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

    draft_roi_img = img[0:int(h*(3/4)), int(w*(3/9)):int(w*(5/8)-20)]
    cv2.imshow("draft_roi_img", draft_roi_img)

    hsv = cv2.cvtColor(draft_roi_img, cv2.COLOR_BGR2HSV)
    h_img, s_img, v_img = cv2.split(hsv)
    #cv2.imshow("h", h_img)
    #cv2.imshow("s", s_img)
    #cv2.imshow("v", v_img)

    marked_img = h_img.copy()

    ret, rng = cv2.threshold(h_img, 80, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("rng",rng)

    # inrange for red number and black number
    #rng = cv2.inRange(img, (100, 100, 100), (255, 255, 255))
    #cv2.imshow("rng",rng)

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
      result = cv2.warpPerspective(draft_roi_img, mtrx, (width, height))
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
      result, semantic_image = self.detect_each_volume(roi_1, 's', 50, semantic_image, pts_each[0])
      if(result == -1):
        error_flag = True
      else:
        # check num_1
        # it should be 0 or 1 only
        if(result != 1):
          result = 0
        vol_array[0] = result

      result, semantic_image = self.detect_each_volume(roi_2, 'v', 90, semantic_image, pts_each[1])
      #print(result)
      if(result == -1):
        error_flag = True
      else:
        vol_array[1] = result

      result, semantic_image = self.detect_each_volume(roi_3, 'v', 90, semantic_image, pts_each[2])
      #print(result)
      if(result == -1):
        error_flag = True
      else:
        vol_array[2] = result

      if(vol_array[1] != 0 or vol_array[2] != 0):
        vol_array[0] = 0

      

    else:
      result, semantic_image = self.detect_each_volume(roi_1, 's', 50, semantic_image, pts_each[0])
      if(result == -1):
        if(self.pre_vol_array[0] != -1):
          result = self.pre_vol_array[0]
          #print(self.pre_vol_array[0])
          vol_array[0] = result
        error_flag = True
      else:
        # check num_1
        # it should be 0 or 1 only
        if(self.pre_vol_array[1] != 9):
          result = 0
        if(result != 1):
          result = 0
          
        vol_array[0] = result

      result, semantic_image = self.detect_each_volume(roi_2, 'v', 90, semantic_image, pts_each[1])
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
          

      result, semantic_image = self.detect_each_volume(roi_3, 'v', 90, semantic_image, pts_each[2])
      #print(result)
      if(result == -1):
        error_flag = True
      else:
        # check num_3
        # it should be +1 or -1 from previous value
        if(self.pre_vol_array[2] == 9):
          if(self.state == 1):
            if(result != 0 and result != 9):
              result = -1
              error_flag = True
          elif(self.state == 2):
            if(result != 9 and result != 8):
              result = -1
              error_flag = True
        elif(self.pre_vol_array[2] ==0):
          if(self.state == 1):
            if(result != 1 and result != 0):
              result = -1
              error_flag = True
          if(self.state == 2):
            if(result != 0 and result != 9):
              result = -1
              error_flag = True
        else:
          if(self.state == 1):
            if(result < self.pre_vol_array[2]):
              result = -1
              error_flag = True
          if(self.state == 2):
            if(result > self.pre_vol_array[2]):
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

  def detect_each_volume(self, roi, format, thresh_value, semantic_image, pt_each):
    result_val = None
    roi_h, roi_w, roi_c = roi.shape

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    hsv_img = None
    bin_img = None
    if(format == 'h'):
      hsv_img = h.copy()
      #cv2.imshow("h", h)
    elif(format == 's'):
      hsv_img = s.copy()
      #cv2.imshow("s", s)
      (thresh, bin_img) = cv2.threshold(hsv_img, thresh_value, 255, cv2.THRESH_BINARY)
      #cv2.imshow("bin", bin_img)
    elif(format == 'v'):
      hsv_img = v.copy()
      #cv2.imshow("v", v)
      (thresh, bin_img) = cv2.threshold(hsv_img, thresh_value, 255, cv2.THRESH_BINARY_INV)
      #cv2.imshow("bin", bin_img)
    
    #(thresh, bin_img) = cv2.threshold(hsv_img, thresh_value, 255, cv2.THRESH_BINARY_INV)
    
    #cv2.waitKey(100)

    contours,hierarchy = cv2.findContours(bin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
        


class NoImageError(Exception):
  """there is no image"""
  pass

class ImageSizeError(Exception):
    def __init__(self, w, h):
      self.msg = "size of image is not 640x480. the size of image is " + str(w) + "x" + str(h) + "."
    def __str__(self):
      return self.msg

class ColorImageError(Exception):
    def __init__(self, c):
      self.msg = "channel of image is not 3(rgb color image). the channel of image is " + str(c) + "."
    def __str__(self):
      return self.msg

class MonoImageError(Exception):
    def __init__(self, c):
      self.msg = "channel of image is not 1(mono gray image). the channel of image is " + str(c) + "."
    def __str__(self):
      return self.msg


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



