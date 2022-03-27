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
    ## marker image publisher. the topic message is "marker_image".
    self.marker_img_pub = rospy.Publisher("marker_image",Image, queue_size=1)
    ## perspective transform image publisher. the topic message is "perspective_transform_image".
    self.trans_img_pub = rospy.Publisher("perspective_transform_image",Image, queue_size=1)
    ## small hsv image(in hsv format) publisher. the topic message is "small_hsv_image".
    self.small_hsv_img_pub = rospy.Publisher("small_hsv_img",Image, queue_size=1)
    ## small h image(in hsv format) publisher. the topic message is "small_h_image".
    self.small_h_img_pub = rospy.Publisher("small_h_img",Image, queue_size=1)
    ## small s image(in hsv format) publisher. the topic message is "small_s_image".
    self.small_s_img_pub = rospy.Publisher("small_s_img",Image, queue_size=1)
    ## small v image(in hsv format) publisher. the topic message is "small_v_image".
    self.small_v_img_pub = rospy.Publisher("small_v_img",Image, queue_size=1)
    ## small binary image publisher. the topic message is "small_binary_image".
    self.small_bin_img_pub = rospy.Publisher("small_binary_img",Image, queue_size=1)
    ## small binary image publisher. the topic message is "small_binary_image".
    self.big_ranged_img_pub = rospy.Publisher("big_ranged_img",Image, queue_size=1)
    ## each_roi_1 image publisher. the topic message is "each_roi_1_image".
    self.each_roi_img_1_pub = rospy.Publisher("each_roi_image_1",Image, queue_size=1)
    ## each_roi_2 image publisher. the topic message is "each_roi_2_image".
    self.each_roi_img_2_pub = rospy.Publisher("each_roi_image_2",Image, queue_size=1)
    ## each_roi_3 image publisher. the topic message is "each_roi_3_image".
    self.each_roi_img_3_pub = rospy.Publisher("each_roi_image_3",Image, queue_size=1)
    ## shadow removed image publisher. the topic message is "shadow_removed_image".
    self.each_shadow_removed_img_1_pub = rospy.Publisher("shadow_removed_image_1",Image, queue_size=1)
    ## shadow removed image publisher. the topic message is "shadow_removed_image".
    self.each_shadow_removed_img_2_pub = rospy.Publisher("shadow_removed_image_2",Image, queue_size=1)
    ## shadow removed image publisher. the topic message is "shadow_removed_image".
    self.each_shadow_removed_img_3_pub = rospy.Publisher("shadow_removed_image_3",Image, queue_size=1)
    ## background removed image publisher. the topic message is "background_removed_image".
    self.each_bg_removed_img_1_pub = rospy.Publisher("background_removed_image_1",Image, queue_size=1)
    ## background removed image publisher. the topic message is "background_removed_image".
    self.each_bg_removed_img_2_pub = rospy.Publisher("background_removed_image_2",Image, queue_size=1)
    ## background removed image publisher. the topic message is "background_removed_image".
    self.each_bg_removed_img_3_pub = rospy.Publisher("background_removed_image_3",Image, queue_size=1)
    ## sharpening image publisher. the topic message is "sharpening_image".
    self.each_sharp_img_1_pub = rospy.Publisher("sharpening_image_1",Image, queue_size=1)
    ## sharpening image publisher. the topic message is "sharpening_image".
    self.each_sharp_img_2_pub = rospy.Publisher("sharpening_image_2",Image, queue_size=1)
    ## sharpening image publisher. the topic message is "sharpening_image".
    self.each_sharp_img_3_pub = rospy.Publisher("sharpening_image_3",Image, queue_size=1)
    ## each binary image publisher. the topic message is "each_binary_image".
    self.each_bin_img_1_pub = rospy.Publisher("each_binary_image_1",Image, queue_size=1)
    ## each binary image publisher. the topic message is "each_binary_image".
    self.each_bin_img_2_pub = rospy.Publisher("each_binary_image_2",Image, queue_size=1)
    ## each binary image publisher. the topic message is "each_binary_image".
    self.each_bin_img_3_pub = rospy.Publisher("each_binary_image_3",Image, queue_size=1)
    ## each morpholgy(close) image publisher. the topic message is "each_close_image".
    self.each_close_img_1_pub = rospy.Publisher("each_close_image_1",Image, queue_size=1)
    ## each morpholgy(close) image publisher. the topic message is "each_close_image".
    self.each_close_img_2_pub = rospy.Publisher("each_close_image_2",Image, queue_size=1)
    ## each morpholgy(close) image publisher. the topic message is "each_close_image".
    self.each_close_img_3_pub = rospy.Publisher("each_close_image_3",Image, queue_size=1)
    ## each noise removed image publisher. the topic message is "each_noise_removed_image".
    self.each_noise_removed_img_1_pub = rospy.Publisher("each_noise_removed_image_1",Image, queue_size=1)
    ## each noise removed image publisher. the topic message is "each_noise_removed_image".
    self.each_noise_removed_img_2_pub = rospy.Publisher("each_noise_removed_image_2",Image, queue_size=1)
    ## each noise removed image publisher. the topic message is "each_noise_removed_image".
    self.each_noise_removed_img_3_pub = rospy.Publisher("each_noise_removed_image_3",Image, queue_size=1)
    ## each mask image publisher. the topic message is "each_mask_image".
    self.each_mask_img_1_pub = rospy.Publisher("each_mask_image_1",Image, queue_size=1)
    ## each mask image publisher. the topic message is "each_mask_image".
    self.each_mask_img_2_pub = rospy.Publisher("each_mask_image_2",Image, queue_size=1)
    ## each mask image publisher. the topic message is "each_mask_image".
    self.each_mask_img_3_pub = rospy.Publisher("each_mask_image_3",Image, queue_size=1)
    ## each small image publisher. the topic message is "each_small_image".
    self.each_small_img_1_pub = rospy.Publisher("each_small_image_1",Image, queue_size=1)
    ## each small image publisher. the topic message is "each_small_image".
    self.each_small_img_2_pub = rospy.Publisher("each_small_image_2",Image, queue_size=1)
    ## each small image publisher. the topic message is "each_small_image".
    self.each_small_img_3_pub = rospy.Publisher("each_small_image_3",Image, queue_size=1)
    ## it is real-time volume publisher
    self.rt_vol_pub = rospy.Publisher("realtime_volume",Int16MultiArray, queue_size=1)
    ## it is real-time volume publisher
    self.valid_vol_pub = rospy.Publisher("valid_volume",Int8, queue_size=1)
    ## semantic image publisher. which have various information
    self.semantic_img_pub = rospy.Publisher("semantic_image",Image, queue_size=1)
    ## it is center difference percentage publisher
    self.center_diff_percentage_pub = rospy.Publisher("center_difference_percentage",Int8, queue_size=1)

    # Variables
    ## self.debug_flag: if debug_flag is true, all debug topic message would published. if debug_flag is false, only necessary topic message would published.
    self.debug_flag = False
    ## we use this member variable when approach to dataset
    self.num = 0
    ## detected real-time volume
    self.rt_vol_msg = Int16MultiArray()
    self.rt_vol_msg.data = [-1, -1, -1]
    ## we use this variable when we visualize the real-time volume 
    self.str_rt_vol = [None, None, None]
    ## integer real-time volume
    self.int_rt_vol = -1
    ## detected valid volume
    self.valid_vol_msg = Int8()
    self.valid_vol_msg.data = -1
    ## list which will append volume
    self.vol_list = []
    ## we use this variable when we visualize the real-time volume 
    self.str_valid_vol = ['X', 'X', 'X']
    ## trained model directory
    model_dir = '/root/catkin_ws/src/ocr/model/knn_trained_model.xml'
    ## knn model
    self.model = cv2.ml.KNearest_create()
    self.model = cv2.ml.KNearest_load(model_dir)
    ## center percentage
    self.center_percentage = 0
    ## center different percentage message
    self.center_diff_percentage_msg = Int8()


  def callback(self,data):
    self.result = [-1, -1, -1]
    try:
      self.estimate_volume(data)

    except CvBridgeError as e:
      print(e)

  def estimate_volume(self, data):
    """!
    @brief we do all process in this function
    @details 

    @param[in] data: ros image message
    
    @note input constraints: 
    @n - none

    @note output constraints: 
    @n - none

    @return none
    """
    cv_image = self.convert_to_cv_image(data)
    if(cv_image is None): return
    else:
      if(self.debug_flag):
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    draft_roi_img = self.get_draft_roi(cv_image)
    if(draft_roi_img is None): return
    else:
      if(self.debug_flag):
        self.draft_roi_img_pub.publish(self.bridge.cv2_to_imgmsg(draft_roi_img, "bgr8"))

    big_hsv_img, big_h_img, big_s_img, big_v_img = self.get_hsv(draft_roi_img)
    if((big_h_img is None) or (big_s_img is None) or (big_v_img is None)): return
    else:
      if(self.debug_flag):
        self.big_hsv_img_pub.publish(self.bridge.cv2_to_imgmsg(big_hsv_img, "bgr8"))
        self.big_h_img_pub.publish(self.bridge.cv2_to_imgmsg(big_h_img, "mono8"))
        self.big_s_img_pub.publish(self.bridge.cv2_to_imgmsg(big_s_img, "mono8"))
        self.big_v_img_pub.publish(self.bridge.cv2_to_imgmsg(big_v_img, "mono8"))

    big_ranged_img = self.get_range(big_hsv_img, 0,100,0,40,255,255)
    if(big_ranged_img is None): return
    else:
      if(self.debug_flag):
        self.big_ranged_img_pub.publish(self.bridge.cv2_to_imgmsg(big_ranged_img, "mono8"))

    big_bin_img = self.get_binary(big_h_img, 40)
    if(big_bin_img is None): return
    else:
      if(self.debug_flag):
        self.big_bin_img_pub.publish(self.bridge.cv2_to_imgmsg(big_bin_img, "mono8"))

    big_noise_removed_bin_img, big_mask_bin_img = self.remove_noise(big_ranged_img, 800, 2000, 20, 60, 20, 60, use_morph = False)
    if(big_noise_removed_bin_img is None): return
    else:
      if(self.debug_flag):
        self.big_noise_removed_bin_img_pub.publish(self.bridge.cv2_to_imgmsg(big_noise_removed_bin_img, "mono8"))
        self.big_mask_bin_img_pub.publish(self.bridge.cv2_to_imgmsg(big_mask_bin_img, "mono8"))

    marker_img, marker_pts = self.detect_marker(big_noise_removed_bin_img)
    if(marker_img is None): return
    else:
      if(self.debug_flag):
        self.marker_img_pub.publish(self.bridge.cv2_to_imgmsg(marker_img, "bgr8"))

    trans_img = self.perspective_transform(draft_roi_img, marker_pts)
    if(trans_img is None): return
    else:
      if(self.debug_flag):
        self.trans_img_pub.publish(self.bridge.cv2_to_imgmsg(trans_img, "bgr8"))

    each_roi_img_1, each_roi_img_2, each_roi_img_3, each_roi_pts = self.get_each_roi(trans_img)
    if((each_roi_img_1 is None) or (each_roi_img_2 is None) or (each_roi_img_3 is None)): return
    else:
      if(self.debug_flag):
        self.each_roi_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_roi_img_1, "bgr8"))
        self.each_roi_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_roi_img_2, "bgr8"))
        self.each_roi_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_roi_img_3, "bgr8"))

    each_shadow_removed_img_1 = self.remove_shadow(each_roi_img_1)
    if(each_shadow_removed_img_1 is None): return
    else:
      if(self.debug_flag):
        self.each_shadow_removed_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_shadow_removed_img_1, "bgr8"))
      
    each_shadow_removed_img_2 = self.remove_shadow(each_roi_img_2)
    if(each_shadow_removed_img_1 is None): return
    else:
      if(self.debug_flag):
        self.each_shadow_removed_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_shadow_removed_img_2, "bgr8"))

    each_shadow_removed_img_3 = self.remove_shadow(each_roi_img_3)
    if(each_shadow_removed_img_1 is None): return
    else:
      if(self.debug_flag):
        self.each_shadow_removed_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_shadow_removed_img_3, "bgr8"))

    each_bg_removed_img_1 = self.remove_background(each_shadow_removed_img_1)
    if(each_bg_removed_img_1 is None): return
    else:
      if(self.debug_flag):
        self.each_bg_removed_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_bg_removed_img_1, "bgr8"))
      
    each_bg_removed_img_2 = self.remove_background(each_shadow_removed_img_2)
    if(each_bg_removed_img_2 is None): return
    else:
      if(self.debug_flag):
        self.each_bg_removed_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_bg_removed_img_2, "bgr8"))

    each_bg_removed_img_3 = self.remove_background(each_shadow_removed_img_3)
    if(each_bg_removed_img_3 is None): return
    else:
      if(self.debug_flag):
        self.each_bg_removed_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_bg_removed_img_3, "bgr8"))

    each_bin_img_1 = self.get_binary(each_bg_removed_img_1, 140)
    if(each_bin_img_1 is None): return
    else:
      if(self.debug_flag):
        self.each_bin_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_bin_img_1, "mono8"))
      
    each_bin_img_2 = self.get_binary(each_bg_removed_img_2, 140)
    if(each_bin_img_2 is None): return
    else:
      if(self.debug_flag):
        self.each_bin_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_bin_img_2, "mono8"))

    each_bin_img_3 = self.get_binary(each_bg_removed_img_3, 140)
    if(each_bin_img_3 is None): return
    else:
      if(self.debug_flag):
        self.each_bin_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_bin_img_3, "mono8"))

    each_noise_removed_img_1, each_mask_img_1 = self.remove_noise(each_bin_img_1, 100, 2000, 0, 40, 0, 40, use_morph = False)
    if(each_noise_removed_img_1 is None): return
    else:
      if(self.debug_flag):
        self.each_noise_removed_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_noise_removed_img_1, "mono8"))
        self.each_mask_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_mask_img_1, "mono8"))
      
    each_noise_removed_img_2, each_mask_img_2 = self.remove_noise(each_bin_img_2, 100, 2000, 0, 40, 0, 40, use_morph = False)
    if(each_noise_removed_img_2 is None): return
    else:
      if(self.debug_flag):
        self.each_noise_removed_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_noise_removed_img_2, "mono8"))
        self.each_mask_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_mask_img_2, "mono8"))

    each_noise_removed_img_3, each_mask_img_3 = self.remove_noise(each_bin_img_3, 100, 2000, 0, 40, 0, 40, use_morph = False)
    if(each_noise_removed_img_3 is None): return
    else:
      if(self.debug_flag):
        self.each_noise_removed_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_noise_removed_img_3, "mono8"))
        self.each_mask_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_mask_img_3, "mono8"))

    bounding_rect_list = []

    each_small_img_1, bounding_rect_1 = self.get_small_image(each_noise_removed_img_1)
    if(each_small_img_1 is None):
      self.rt_vol_msg.data[0] = -1
      self.str_rt_vol[0] = "X"
    else:
      if(self.debug_flag):
        self.each_small_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_small_img_1, "mono8"))
    bounding_rect_list.append(bounding_rect_1)

    each_small_img_2, bounding_rect_2 = self.get_small_image(each_noise_removed_img_2)
    if(each_small_img_2 is None): 
      self.rt_vol_msg.data[1] = -1
      self.str_rt_vol[1] = "X"
    else:
      if(self.debug_flag):
        self.each_small_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_small_img_2, "mono8"))
    bounding_rect_list.append(bounding_rect_2)

    each_small_img_3, bounding_rect_3 = self.get_small_image(each_noise_removed_img_3)
    if(each_small_img_3 is None):
      self.rt_vol_msg.data[2] = -1
      self.str_rt_vol[2] = "X"
    else:
      if(self.debug_flag):
        self.each_small_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_small_img_3, "mono8"))
    bounding_rect_list.append(bounding_rect_3)



    # name = '/root/catkin_ws/src/ocr/dataset/9/' + str(self.num) + '.jpg'
    # cv2.imwrite(name, each_small_img_3)

    # self.num = self.num + 1
    # if(self.num > 1000):

    #   self.num = 1000

    vol_1 = self.detect_volume(each_small_img_1)
    if(vol_1 == -1):
      self.rt_vol_msg.data[0] = -1
      self.str_rt_vol[0] = "X"
    else:
      self.rt_vol_msg.data[0] = vol_1
      self.str_rt_vol[0] = str(vol_1)

    vol_2 = self.detect_volume(each_small_img_2)
    if(vol_2 == -1):
      self.rt_vol_msg.data[1] = -1
      self.str_rt_vol[1] = "X"
    else:
      self.rt_vol_msg.data[1] = vol_2
      self.str_rt_vol[1] = str(vol_2)

    vol_3 = self.detect_volume(each_small_img_3)
    if(vol_3 == -1):
      self.rt_vol_msg.data[2] = -1
      self.str_rt_vol[2] = "X"
    else:
      self.rt_vol_msg.data[2] = vol_3
      self.str_rt_vol[2] = str(vol_3)

    if(self.debug_flag):
      self.rt_vol_pub.publish(self.rt_vol_msg)

    if((vol_1 != -1) and (vol_2 != -1) and (vol_3 != -1)):
      self.int_rt_vol = vol_1*100 + vol_2*10 + vol_3
    else:
      self.int_rt_vol = -1
    
    if(self.int_rt_vol != -1):
      self.valid_vol_msg.data = self.get_valid_volume(self.int_rt_vol, max_param = 1)
      
      self.valid_vol_pub.publish(self.valid_vol_msg)

    semantic_img, center_diff_percentage = self.get_semantic_image(trans_img, each_roi_pts, bounding_rect_list, self.str_rt_vol, self.valid_vol_msg.data)
    if (semantic_img is None): return
    else:
      self.semantic_img_pub.publish(self.bridge.cv2_to_imgmsg(semantic_img, "bgr8"))
      self.center_diff_percentage_msg.data = center_diff_percentage
      self.center_diff_percentage_pub.publish(self.center_diff_percentage_msg)

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
        raise InputImageSizeError(w, h)

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
        raise InputImageSizeError(w, h)
      if (c != 3):
        raise ColorImageError(c)

      # doing
      draft_roi_img = img[int(h*(0/20)):int(h*(16/20)), int(w*(6/20)):int(w*(13/20))]

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

  def get_binary(self, img, thr, use_inv = True):
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
      if(len(img.shape)==3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # input error check
      if img is None:
        raise NoImageError
      if (img.dtype != 'uint8'):
        raise MonoImageError(c)

      # doing
      #blur = cv2.GaussianBlur(img,(11,11),0)
      bin_img = None
      if(use_inv):
        ret, bin_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY_INV)
      else:
        ret, bin_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
      #ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

      cv2.imshow("bin_img", bin_img)
      #cv2.waitKey()

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

  def remove_noise(self, img, area_min, area_max, w_min, w_max, h_min, h_max, use_morph = True):
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
        raise MonoImageError(img.dtype)

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
        if cv2.contourArea(c)<area_min or cv2.contourArea(c)>area_max \
          or h<h_min or h>h_max or w<w_min or w>w_max:
          cv2.drawContours(mask, [c], -1, 0, -1)
        # else:
        #   print(cv2.contourArea(c))
        #   print(w)
        #   print(h)
        
      # remove the contours from the image and show the resulting images
      noise_removed_bin_img = cv2.bitwise_and(img, img, mask=mask)

      if(use_morph):
        # somtimes, bad noise make the marker uncertain. to prevent that issue we adapt morphology method to it.
        kernel = np.ones((15, 15), np.uint8)
        noise_removed_bin_img = cv2.morphologyEx(noise_removed_bin_img, cv2.MORPH_OPEN, kernel)

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

  def detect_marker(self, img):
    """!
    @brief detect marker and return the points of marker.
    @details

    @param[in] img: binary image which is drawn the marker area as white
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be binary image (1 channel)

    @note output constraints: 
    @n - cnts_img should not be empty 
    @n - cnts_img should be color image (3 channel)
    @n - number of cnts should be 4

    @return makered_img: output marker image which is drawn contour and moment.
    @return cnts: contour which is detected from binary image. the number of contour should be 4.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.dtype != 'uint8'):
        raise MonoImageError(img.dtype)

      # doing
      marked_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

      pts = np.zeros((4, 2), dtype=np.float32)
      cnts, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      if(len(cnts) == 4):
        for i in range(len(cnts)):
          M = cv2.moments(cnts[i])
          cX = int(M['m10'] / M['m00'])
          cY = int(M['m01'] / M['m00'])
          
          cv2.circle(marked_img, (cX, cY), 3, (255, 0, 0), -1)
          cv2.drawContours(marked_img, [cnts[i]], 0, (0, 0, 255), 2)

          pts[i] = [cX, cY]

      # output error check
      if (marked_img is None):
        raise NoImageError
      if (marked_img.shape[2] != 3):
        raise ColorImageError(marked_img.shape[2])
      if (len(pts) != 4):
        raise MarkerNumberError(len(pts))

      # return
      return marked_img, pts

    except Exception as e:
      print(e)
      return None, None
    pass

  def perspective_transform(self, img, pts):
    """!
    @brief based on detected marker, conduct perspective transform
    @details it's refered from "https://minimin2.tistory.com/135"

    @param[in] img: draft roi image. it should be color image.
    @param[in] pts: points of markers. the number of points should be 4.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)
    @n - the number of points should be 4

    @note output constraints: 
    @n - trans_img should not be empty 
    @n - trans_img should be color image (3 channel)

    @return trans_img: output perspective transform image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.shape[2] != 3):
        raise ColorImageError(img.shape[2])
      if (len(pts) != 4):
        raise MarkerNumberError(len(pts))
      
      # doing
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
      trans_img = cv2.warpPerspective(img, mtrx, (width, height))

      # output error check
      if (trans_img is None):
        raise NoImageError
      if (trans_img.shape[2] != 3):
        raise ColorImageError(trans_img.shape[2])
      
      # return
      return trans_img

    except Exception as e:
      print(e)
      return None
    pass

  def get_range(self, img, min1, min2, min3, max1, max2, max3):
    """!
    @brief for rgb or hsv image we could adapt inrange function in opencv
    @details

    @param[in] img: input image. it should be color image.
    @param[in] min1: the first min parameter.
    @param[in] min2: the second min parameter.
    @param[in] min3: the thirds min parameter.
    @param[in] max1: the first max parameter.
    @param[in] max2: the second max parameter.
    @param[in] max3: the thirds max parameter.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)
    @n - all min~max values should be ranged in 0~255

    @note output constraints: 
    @n - ranged_img should not be empty 
    @n - ranged_img should be binary image (1 channel)

    @return ranged_img: output ranged image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.shape[2] != 3):
        raise ColorImageError(img.shape[2])
      if (min1 < 0 or min1 > 255):
        raise RangeError(min1)
      if (min2 < 0 or min2 > 255):
        raise RangeError(min2)
      if (min3 < 0 or min3 > 255):
        raise RangeError(min3)
      if (max1 < 0 or max1 > 255):
        raise RangeError(max1)
      if (max2 < 0 or max2 > 255):
        raise RangeError(max2)
      if (max3 < 0 or max3 > 255):
        raise RangeError(max3)
      if (min1 > max1):
        raise MinMaxError(min1, max1)
      if (min2 > max2):
        raise MinMaxError(min2, max2)
      if (min3 > max3):
        raise MinMaxError(min3, max3)
      
      # doing
      ranged_img = cv2.inRange(img, (min1, min2, min3), (max1, max2, max3))

      # output error check
      if (ranged_img is None):
        raise NoImageError
      if (ranged_img.dtype != 'uint8'):
        raise MonoImageError(ranged_img.dtype)
      
      # return
      return ranged_img

    except Exception as e:
      print(e)
      return None
    pass

  def get_each_roi(self, img):
    """!
    @brief based on perspective transform image, we get each digit of volume indicator image in vaild roi
    @details

    @param[in] img: perspective transform image. it should be color image
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - roi_1 should not be empty 
    @n - roi_1 should be color image (3 channel)
    @n - roi_2 should not be empty 
    @n - roi_2 should be color image (3 channel)
    @n - roi_3 should not be empty 
    @n - roi_3 should be color image (3 channel)

    @return roi_1: first digit image
    @return roi_2: second digit image
    @return roi_3: third digit image
    """
    # input error check
    if img is None:
      raise NoImageError
    if (img.shape[2] != 3):
      raise ColorImageError(img.shape[2])

    # doing
    pts_each = [[],[],[]]

    x=45; y=65; w=40; h=40
    roi_1 = img[y:y+h, x:x+w]

    pts_each[0].append(x)
    pts_each[0].append(y)

    x=45; y=125; w=40; h=40
    roi_2 = img[y:y+h, x:x+w]

    pts_each[1].append(x)
    pts_each[1].append(y)

    x=45; y=185; w=40; h=40
    roi_3 = img[y:y+h, x:x+w]

    pts_each[2].append(x)
    pts_each[2].append(y)

    # output error check
    try:
      if (roi_1 is None) or (roi_2 is None) or (roi_3 is None):
        raise NoImageError
      if (roi_1.shape[2] != 3) or (roi_1.shape[2] != 3) or (roi_1.shape[2] != 3):
        raise ColorImageError(roi_1.shape[2])
      if ((bool(pts_each[0]) == False) or (bool(pts_each[1]) == False) or (bool(pts_each[2]) == False)):
        raise EmptyListError()

      # return
      return roi_1, roi_2, roi_3, pts_each
    except Exception as e:
      print(e)
      return None, None, None, None
    pass

  def remove_shadow(self, img):
    """!
    @brief when we detec digit of volume indicator, there are too many shadow noise that we don't need. so we remove shadow.
    @details

    @param[in] img: input image. it should be color image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - shadow_removed_img should not be empty 
    @n - shadow_removed_img should be color image (3 channel)

    @return shadow_removed_img: output shadow removed image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.shape[2] != 3):
        raise ColorImageError(img.shape[2])
      
      # doing
      rgb_planes = cv2.split(img)
      result_norm_planes = []

      for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

      shadow_removed_img = cv2.merge(result_norm_planes)

      
      
      # output error check
      if (shadow_removed_img is None):
        raise NoImageError
      if (shadow_removed_img.shape[2] != 3):
        raise ColorImageError(shadow_removed_img.shape[2])
      
      # return
      return shadow_removed_img

    except Exception as e:
      print(e)
      return None
    pass

  def get_sharpening(self, img):
    """!
    @brief process sharpening for detecting digit more accurately
    @details

    @param[in] img: input image. it should be color image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - sharp_img should not be empty 
    @n - sharp_img should be color image (3 channel)

    @return sharp_img: output sharpening image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.shape[2] != 3):
        raise ColorImageError(img.shape[2])
      
      # doing
      sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
      sharp_img = cv2.filter2D(img, -1, sharpening_mask)


          
      # output error check
      if (sharp_img is None):
        raise NoImageError
      if (sharp_img.shape[2] != 3):
        raise ColorImageError(sharp_img.shape[2])
      
      # return
      return sharp_img

    except Exception as e:
      print(e)
      return None
    pass



  def remove_background(self, img):
    """!
    @brief remove background and leave only digit of micropipette volume indicator
    @details

    @param[in] img: input image. it should be color image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - background_removed_img should not be empty 
    @n - background_removed_img should be color image (3 channel)

    @return background_removed_img: output background removed image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.shape[2] != 3):
        raise ColorImageError(img.shape[2])
      
      # doing      
      bg_removed_img = img.copy()
      
      h, w, c = img.shape
      seed = (1, 1)

      # Use floodFill for filling the background with black color
      cv2.floodFill(bg_removed_img, None, seedPoint=seed, newVal=(255, 255, 255), loDiff=(20, 20, 20, 20), upDiff=(100, 100, 100, 100))

      # output error check
      if (bg_removed_img is None):
        raise NoImageError
      if (bg_removed_img.shape[2] != 3):
        raise ColorImageError(bg_removed_img.shape[2])
      
      # return
      return bg_removed_img

    except Exception as e:
      print(e)
      return None
    pass

  def get_small_image(self, img):
    """!
    @brief we use small image(10x10) to adapt to pre-trained model. so we convert image in here.
    @details

    @param[in] img: binary input image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be binary image (1 channel)

    @note output constraints: 
    @n - small_img should not be empty 
    @n - size of small_img should be 10x10
    @n - smal_img should be gray image (1 channel)

    @return : output small size image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.dtype != 'uint8'):
        raise MonoImageError(img.dtype)
      
      # doing
      # cv2.imshow("input_img", img)
      img_h, img_w = img.shape
      contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

      #print(len(contours))
      small_img = None
      bounding_rect = []
      if(len(contours) == 1):
        cnt = contours[0]
        [x,y,w,h] = cv2.boundingRect(cnt)
        bounding_rect = [x,y,w,h]
        M = cv2.moments(cnt, False)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(x, y, w, h)
        
        print(cx)
        print(int(img_w*(1/4)))
        if((cx < int(img_w*(1/4))) or (cx > int(img_w*(3/4)))):
          raise CenterError(cx, cy)

        if(w >= h):
          h = w
          roi_bin = img[cy-int(h/2):cy+int(h/2),x:x+w]
        else:
          w = h
          #print(cx-int(w/2))
          x_start = cx-int(w/2)
          if(x_start < 0):
            x_start = 0
          roi_bin = img[y:y+h,x_start:cx+int(w/2)]
        
        #print(roi_bin.shape)
        if((roi_bin.shape[0] == 0) or roi_bin.shape[1] == 0):
          raise ResizeError(roi_bin.shape[0], roi_bin.shape[1])
        small_img = cv2.resize(roi_bin,(10,10))
        
      # output error check
      if (len(contours) != 1):
        raise ContourError(len(contours))
      if (small_img is None):
        raise NoImageError
      h, w = small_img.shape
      if ((h != 10) or (w != 10)):
        raise SmallImageSizeError(w, h)
      if (small_img.dtype != 'uint8'):
        raise MonoImageError(small_img.dtype)
      
      # return
      return small_img, bounding_rect

    except Exception as e:
      print(e)
      return None, None
    pass

  def detect_volume(self, img):
    """!
    @brief finally, we detect number using pre-train model
    @details

    @param[in] img: binary input image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image size should be 10x10
    @n - image should be binary image (1 channel)

    @note output constraints: 
    @n - image should not be empty 
    @n - image size should be 10x10
    @n - image should be binary image (1 channel)

    @return vol: output detected volume.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      h, w = img.shape
      if ((h != 10) or (w != 10)):
        raise SmallImageSizeError(w, h)
      if (img.dtype != 'uint8'):
        raise MonoImageError(img.dtype)
      
      # doing
      roismall = img.reshape((1,100))
      roismall = np.float32(roismall)
          
      retval, results, neigh_resp, dists = self.model.findNearest(roismall, k = 3)
      
      if (dists[0][0] > 500000):
            raise KnnDistError(retval, dists[0][0])

      # debug_string = "retval: " + str(retval) + " / results: " + str(results) + " / neigh_resp: " + str(neigh_resp) + " / dists: " + str(dists)
      # print(debug_string)

      vol = int(retval)

      # output error check
      if(vol < 0 or vol > 9):
        raise EachVolumeError(vol)

      # return
      return vol

    except Exception as e:
      print(e)
      return -1
    pass

  def get_valid_volume(self, rt_vol, max_param = 5):
    """!
    @brief we calculate valid volume with max function.
    @details the method is simple. we append real-time volume to list and if the list is bigger than max_param then it will get max volume.

    @param[in] rt_vol: real-time volume
    @param[in] vol_list: volume list, it is member variable
    @param[in] max_param: it would decide the volumes of list is valid
    
    @note input constraints: 
    @n - rt_vol should be ranged 0~100
    @n - vol_list should not be over max_param

    @note output constraints: 
    @n - valid_vol should be ranged 0~100

    @return valid_vol: valid volume
    """
    try:
      # input error check
      if rt_vol < 0 or rt_vol > 100:
        raise VolumeError(rt_vol)
      if (len(self.vol_list) > max_param):
        raise VolumeListError(len(self.vol_list), max_param)
      
      # doing
      self.vol_list.append(rt_vol)
      if(len(self.vol_list) > max_param):
        self.vol_list.pop(0)
        valid_vol = max(self.vol_list)
      else:
        raise NotEnoughVolumeListError(len(self.vol_list), max_param)

      # output error check
      if valid_vol < 0 or valid_vol > 100:
        raise VolumeError(valid_vol)

      # return
      return valid_vol

    except Exception as e:
      print(e)
      return None
    pass

  def create_text_data(self):
    """!
    @brief create text data
    @details

    @param[in] img: binary input image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image size should be 40x40 (size of dataset)
    @n - image should be binary image (1 channel)
    """
    try:
      
      num_dir = 0
      samples =  np.empty((0,100))
      responses = []
      while num_dir < 10:
        num = 0
        while num < 1000:
          name = '/root/catkin_ws/src/ocr/dataset/'+ str(num_dir) + '/' + str(num) + '.jpg'
          img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
          # cv2.imshow("img",img)
          # cv2.waitKey(100)

          # input error check
          if img is None:
            raise NoImageError
          h, w = img.shape
          if ((h != 10) or (w != 10)):
            raise SmallImageSizeError(w, h)
          if (img.dtype != 'uint8'):
            raise MonoImageError(img.dtype)
      
          # doing
          responses.append(int(num_dir))
          sample = img.reshape((1,100))
          samples = np.append(samples,sample,0)

          num = num+1
          string = "process: " + str(num)
          print(string)
        num_dir = num_dir+1
      responses = np.array(responses,np.float32)
      responses = responses.reshape((responses.size,1))

      print("create text data complete")

      np.savetxt('/root/catkin_ws/src/ocr/dataset/generalsamples.data',samples)
      np.savetxt('/root/catkin_ws/src/ocr/dataset/generalresponses.data',responses)

    except Exception as e:
      print(e)
    pass

  def ocr_train(self):
    """!
    @brief train
    @details

    @param[in] img: binary input image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image size should be 40x40 (size of dataset)
    @n - image should be binary image (1 channel)
    """
    try:

      samples = np.loadtxt('/root/catkin_ws/src/ocr/dataset/generalsamples.data',np.float32)
      responses = np.loadtxt('/root/catkin_ws/src/ocr/dataset/generalresponses.data',np.float32)
      responses = responses.reshape((responses.size,1))

      model = cv2.ml.KNearest_create()
      model.train(samples,cv2.ml.ROW_SAMPLE,responses)
      save_dir = '/root/catkin_ws/src/ocr/model/knn_trained_model.xml'
      model.save(save_dir)
      
      acc_cnt = 0
      num_dir = 0
      while num_dir < 10:
        num = 0
        while num < 1000:
          name = '/root/catkin_ws/src/ocr/dataset/'+ str(num_dir) + '/' + str(num) + '.jpg'
          img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

          # input error check
          if img is None:
            raise NoImageError
          h, w = img.shape
          if ((h != 10) or (w != 10)):
            raise SmallImageSizeError(w, h)
          if (img.dtype != 'uint8'):
            raise MonoImageError(img.dtype)
      
          # doing
          roismall = img.reshape((1,100))
          roismall = np.float32(roismall)

          retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)

          string = "process: " + str(num_dir) + " / " + str(num) + " is " + str(retval)
          print(string)
          if(num_dir == retval):
            acc_cnt = acc_cnt + 1

          num = num + 1
        num_dir = num_dir + 1
      
      accuracy = acc_cnt / 10000
      string = "accuracy: " + str(accuracy)
      print(string)

    except Exception as e:
      print(e)
    pass

  def get_morpholgy_close(self, img):
    """!
    @brief to eleminate noise of image, we adapt morpholgy(Close)
    @details

    @param[in] img: input image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - morph_img should not be empty 
    @n - morph_img should be gray image (1 channel)

    @return morph_img: output morphology image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.shape[2] != 3):
        raise ColorImageError(img.shape[2])
      
      # doing
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      kernel = np.ones((3, 3), np.uint8)
      morph_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

      # output error check
      if (morph_img is None):
        raise NoImageError
      if (morph_img.dtype != 'uint8'):
        raise MonoImageError(morph_img.dtype)

      # return
      return morph_img

    except Exception as e:
      print(e)
      return None
    pass

  def get_semantic_image(self, img, each_roi_pts, bounding_rect_list, rt_vol_list, valid_vol):
    """!
    @brief draw semantic image for debugging.
    @details it has marker roi image which which has bounding rectangle for each digit of volume indicator.
    @n there are also information like real-time volum / valid volume
    @n real-time volume is estimated volume at every time. even though all the digits are not detected successfully.
    @n valid volume is calculated real-time value. when the digits are detected successfully, we append this volume to list. and 

    @param[in] img: input image. it should be roi image maden with marker
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be color image (3 channel)

    @note output constraints: 
    @n - semantic_img should not be empty 
    @n - semantic_img should be color image (3 channel)

    @return semantic_img: output semantic image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.shape[2] != 3):
        raise ColorImageError(img.shape[2])

      # doing
      semantic_img = img.copy()

      black_img = np.zeros([semantic_img.shape[0],250,3],dtype=np.uint8)
      semantic_img = np.concatenate((semantic_img, black_img), axis=1)
      if(bounding_rect_list[0] != None):
        roi_x_1 = int(each_roi_pts[0][0])
        roi_y_1 = int(each_roi_pts[0][1])
        bounding_rect_x_1 = bounding_rect_list[0][0]
        bounding_rect_y_1 = bounding_rect_list[0][1]
        bounding_rect_w_1 = bounding_rect_list[0][2]
        bounding_rect_h_1 = bounding_rect_list[0][3]
        cv2.rectangle(semantic_img,(roi_x_1+bounding_rect_x_1,roi_y_1+bounding_rect_y_1),(roi_x_1+bounding_rect_x_1+bounding_rect_w_1,roi_y_1+bounding_rect_y_1+bounding_rect_h_1),(0,255,0),2)

      if(bounding_rect_list[1] != None):
        roi_x_2 = int(each_roi_pts[1][0])
        roi_y_2 = int(each_roi_pts[1][1])
        bounding_rect_x_2 = bounding_rect_list[1][0]
        bounding_rect_y_2 = bounding_rect_list[1][1]
        bounding_rect_w_2 = bounding_rect_list[1][2]
        bounding_rect_h_2 = bounding_rect_list[1][3]
        cv2.rectangle(semantic_img,(roi_x_2+bounding_rect_x_2,roi_y_2+bounding_rect_y_2),(roi_x_2+bounding_rect_x_2+bounding_rect_w_2,roi_y_2+bounding_rect_y_2+bounding_rect_h_2),(0,255,0),2)


      roi_x_3 = int(each_roi_pts[2][0])
      roi_y_3 = int(each_roi_pts[2][1])
      cv2.rectangle(semantic_img,(roi_x_3,roi_y_3),(roi_x_3+40,roi_y_3+40),(255,255,0),2)
      cv2.line(semantic_img,(roi_x_3+20,roi_y_3),(roi_x_3+20,roi_y_3+40),(0,0,255),2)
      if(bounding_rect_list[2] != None):
        
        bounding_rect_x_3 = bounding_rect_list[2][0]
        bounding_rect_y_3 = bounding_rect_list[2][1]
        bounding_rect_w_3 = bounding_rect_list[2][2]
        bounding_rect_h_3 = bounding_rect_list[2][3]
        cv2.rectangle(semantic_img,(roi_x_3+bounding_rect_x_3,roi_y_3+bounding_rect_y_3),(roi_x_3+bounding_rect_x_3+bounding_rect_w_3,roi_y_3+bounding_rect_y_3+bounding_rect_h_3),(0,255,0),2)
        # drawing center line
        cx_3 = roi_x_3+bounding_rect_x_3+int(bounding_rect_w_3/2)
        cy_3_top = roi_y_3+bounding_rect_y_3
        cy_3_bottom = roi_y_3+bounding_rect_y_3+bounding_rect_h_3
        cv2.line(semantic_img,(cx_3,cy_3_top),(cx_3,cy_3_bottom),(0,128,255),2)
        
        self.center_percentage = int(((cx_3 - (roi_x_3+20))/20)*100)

      str_valid_vol = str(valid_vol).zfill(3)
      str_valid_vol = "valid volume: " + str_valid_vol
      
      str_rt_vol = rt_vol_list[0]
      str_rt_vol = str_rt_vol + rt_vol_list[1]
      str_rt_vol = str_rt_vol + rt_vol_list[2]
      str_rt_vol = "real-time value: " + str_rt_vol    

      str_center_percentage = str(self.center_percentage)
      str_center_percentage = "center percentage: " + str_center_percentage + "%"

      cv2.putText(semantic_img,str_valid_vol,(130, 20),0,0.5,(255,255,255),1)
      cv2.putText(semantic_img,str_rt_vol,(130, 70),0,0.5,(255,255,255),1)
      cv2.putText(semantic_img,str_center_percentage,(130, 120),0,0.5,(255,255,255),1)

      # output error check
      if (semantic_img is None):
        raise NoImageError
      if (semantic_img.shape[2] != 3):
        raise ColorImageError(semantic_img.shape[2])

      # return
      return semantic_img, self.center_percentage

    except Exception as e:
      print(e)
      return None, None
    pass

class NoImageError(Exception):
  """there is no image"""
  pass

class InputImageSizeError(Exception):
    def __init__(self, w, h):
      self.msg = "size of image is not 640x480. the size of image is " + str(w) + "x" + str(h) + "."
    def __str__(self):
      return self.msg

class ROIImageSizeError(Exception):
    def __init__(self, w, h):
      self.msg = "size of small image is not 40x40. the size of image is " + str(w) + "x" + str(h) + "."
    def __str__(self):
      return self.msg

class SmallImageSizeError(Exception):
    def __init__(self, w, h):
      self.msg = "size of small image is not 10x10. the size of image is " + str(w) + "x" + str(h) + "."
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

class MarkerNumberError(Exception):
    def __init__(self, c):
      self.msg = "number of marker should be 4, but detected marker's number is " + str(c) + "."
    def __str__(self):
      return self.msg

class RangeError(Exception):
    def __init__(self, c):
      self.msg = "the range should be 0~255. but the value is "+ str(c) + "."
    def __str__(self):
      return self.msg

class MinMaxError(Exception):
    def __init__(self, c1, c2):
      self.msg = "min value could not be bigger than max value. min value is  " + str(c1) + "max value is  " + str(c2)
    def __str__(self):
      return self.msg

class EmptyListError(Exception):
    def __init__(self):
      self.msg = "list is empty."
    def __str__(self):
      return self.msg

class ContourError(Exception):
    def __init__(self, c):
      self.msg = "the number contour should be 1. but the number of contour is " + str(c) + "."
    def __str__(self):
      return self.msg

class CenterError(Exception):
    def __init__(self, cx, cy):
      self.msg = "the digit is not located in center area. it's located in " + str(cx) + "x" + str(cy) + "."
    def __str__(self):
      return self.msg

class ResizeError(Exception):
    def __init__(self, w, h):
      self.msg = "size of input image has error value. which are " + str(w) + "x" + str(h) + "."
    def __str__(self):
      return self.msg

class EachVolumeError(Exception):
    def __init__(self, vol):
      self.msg = "range of each volum should be 0~9. but the volum is " + str(vol) + "."
    def __str__(self):
      return self.msg

class VolumeError(Exception):
    def __init__(self, vol):
      self.msg = "range of volum should be 0~100. but the volum is " + str(vol) + "."
    def __str__(self):
      return self.msg

class VolumeListError(Exception):
    def __init__(self, leng_of_vol_list, max_param):
      self.msg = "length of volume list should be smaller than " + str(max_param) + ". but the length of the volume list is " + str(leng_of_vol_list) + "."
    def __str__(self):
      return self.msg

class NotEnoughVolumeListError(Exception):
    def __init__(self, leng_of_vol_list, max_param):
      self.msg = "length of volume list should be at least " + str(max_param) + ". but the length of the volume list is " + str(leng_of_vol_list) + "."
    def __str__(self):
      return self.msg

class KnnDistError(Exception):
    def __init__(self, retval, dist):
      self.msg = "distance of knn sould be at least 500,000 but the dist of "+ str(retval) + " is " + str(dist) + "."
    def __str__(self):
      return self.msg

def main(args):
  rospy.init_node('ocr_node', anonymous=True)
  ocr = OCR()
  try:
    #ocr.create_text_data()
    #ocr.ocr_train()
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)



