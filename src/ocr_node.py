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
    self.ranged_img_pub = rospy.Publisher("ranged_img",Image, queue_size=1)
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

    big_bin_img = self.get_binary(big_h_img, 50)
    if(big_bin_img is None): return

    big_noise_removed_bin_img, big_mask_bin_img = self.remove_noise(big_bin_img, 800, 2000, 20, 60, 20, 60, use_morph = True)
    if(big_noise_removed_bin_img is None): return

    marker_img, marker_pts = self.detect_marker(big_noise_removed_bin_img)
    if(marker_img is None): return

    trans_img = self.perspective_transform(draft_roi_img, marker_pts)
    if(trans_img is None): return

    each_roi_img_1, each_roi_img_2, each_roi_img_3, each_roi_pts = self.get_each_roi(trans_img)
    if((each_roi_img_1 is None) or (each_roi_img_2 is None) or (each_roi_img_3 is None)): return

    each_shadow_removed_img_1 = self.remove_shadow(each_roi_img_1)
    if(each_shadow_removed_img_1 is None): return

    each_shadow_removed_img_2 = self.remove_shadow(each_roi_img_2)
    if(each_shadow_removed_img_1 is None): return

    each_shadow_removed_img_3 = self.remove_shadow(each_roi_img_3)
    if(each_shadow_removed_img_1 is None): return

    each_bg_removed_img_1 = self.remove_background(each_shadow_removed_img_1)
    if(each_bg_removed_img_1 is None): return

    each_bg_removed_img_2 = self.remove_background(each_shadow_removed_img_2)
    if(each_bg_removed_img_2 is None): return

    each_bg_removed_img_3 = self.remove_background(each_shadow_removed_img_3)
    if(each_bg_removed_img_3 is None): return

    each_bin_img_1 = self.get_binary(each_bg_removed_img_1, 140)
    if(each_bin_img_1 is None): return

    each_bin_img_2 = self.get_binary(each_bg_removed_img_2, 140)
    if(each_bin_img_2 is None): return

    each_bin_img_3 = self.get_binary(each_bg_removed_img_3, 140)
    if(each_bin_img_3 is None): return

    each_noise_removed_img_1, each_mask_img_1 = self.remove_noise(each_bin_img_1, 100, 2000, 0, 40, 0, 40, use_morph = False)
    if(each_noise_removed_img_1 is None): return

    each_noise_removed_img_2, each_mask_img_2 = self.remove_noise(each_bin_img_2, 100, 2000, 0, 40, 0, 40, use_morph = False)
    if(each_noise_removed_img_2 is None): return

    each_noise_removed_img_3, each_mask_img_3 = self.remove_noise(each_bin_img_3, 100, 2000, 0, 40, 0, 40, use_morph = False)
    if(each_noise_removed_img_3 is None): return

    each_small_img_1 = self.get_small_image(each_noise_removed_img_1)
    if(each_small_img_1 is None): return

    each_small_img_2 = self.get_small_image(each_noise_removed_img_2)
    if(each_small_img_2 is None): return

    each_small_img_3 = self.get_small_image(each_noise_removed_img_3)
    if(each_small_img_3 is None): return

    # each_close_img_1 = self.get_morpholgy_close(each_bg_removed_img_1)
    # if(each_close_img_1 is None): return

    # each_close_img_2 = self.get_morpholgy_close(each_bg_removed_img_2)
    # if(each_close_img_2 is None): return

    # each_close_img_3 = self.get_morpholgy_close(each_bg_removed_img_3)
    # if(each_close_img_3 is None): return

    # each_sharp_img_1 = self.get_sharpening(each_bg_removed_img_1)
    # if(each_sharp_img_1 is None): return

    # each_sharp_img_2 = self.get_sharpening(each_bg_removed_img_2)
    # if(each_sharp_img_2 is None): return

    # each_sharp_img_3 = self.get_sharpening(each_bg_removed_img_3)
    # if(each_sharp_img_3 is None): return


    # small_hsv_img, small_h_img, small_s_img, small_v_img = self.get_hsv(shadow_removed_img)
    # if((small_h_img is None) or (small_s_img is None) or (small_v_img is None)): return

    # ranged_img = self.get_range(shadow_removed_img,0,0,0,200,200,200)

    # small_bin_img = self.get_binary(shadow_removed_img, 70)
    # if(small_bin_img is None): return

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
      self.marker_img_pub.publish(self.bridge.cv2_to_imgmsg(marker_img, "bgr8"))
      self.trans_img_pub.publish(self.bridge.cv2_to_imgmsg(trans_img, "bgr8"))
      self.each_roi_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_roi_img_1, "bgr8"))
      self.each_roi_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_roi_img_2, "bgr8"))
      self.each_roi_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_roi_img_3, "bgr8"))
      self.each_shadow_removed_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_shadow_removed_img_1, "bgr8"))
      self.each_shadow_removed_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_shadow_removed_img_2, "bgr8"))
      self.each_shadow_removed_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_shadow_removed_img_3, "bgr8"))
      self.each_bg_removed_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_bg_removed_img_1, "bgr8"))
      self.each_bg_removed_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_bg_removed_img_2, "bgr8"))
      self.each_bg_removed_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_bg_removed_img_3, "bgr8"))
      self.each_bin_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_bin_img_1, "mono8"))
      self.each_bin_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_bin_img_2, "mono8"))
      self.each_bin_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_bin_img_3, "mono8"))
      self.each_noise_removed_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_noise_removed_img_1, "mono8"))
      self.each_noise_removed_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_noise_removed_img_2, "mono8"))
      self.each_noise_removed_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_noise_removed_img_3, "mono8"))
      self.each_mask_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_mask_img_1, "mono8"))
      self.each_mask_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_mask_img_2, "mono8"))
      self.each_mask_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_mask_img_3, "mono8"))
      self.each_small_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_small_img_1, "mono8"))
      self.each_small_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_small_img_2, "mono8"))
      self.each_small_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_small_img_3, "mono8"))
      
      # self.each_close_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_close_img_1, "mono8"))
      # self.each_close_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_close_img_2, "mono8"))
      # self.each_close_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_close_img_3, "mono8"))
      # self.small_hsv_img_pub.publish(self.bridge.cv2_to_imgmsg(small_hsv_img, "bgr8"))
      # self.small_h_img_pub.publish(self.bridge.cv2_to_imgmsg(small_h_img, "mono8"))
      # self.small_s_img_pub.publish(self.bridge.cv2_to_imgmsg(small_s_img, "mono8"))
      # self.small_v_img_pub.publish(self.bridge.cv2_to_imgmsg(small_v_img, "mono8"))
      # self.small_bin_img_pub.publish(self.bridge.cv2_to_imgmsg(small_bin_img, "mono8"))
      # self.ranged_img_pub.publish(self.bridge.cv2_to_imgmsg(ranged_img, "mono8"))
      # self.each_sharp_img_1_pub.publish(self.bridge.cv2_to_imgmsg(each_sharp_img_1, "bgr8"))
      # self.each_sharp_img_2_pub.publish(self.bridge.cv2_to_imgmsg(each_sharp_img_2, "bgr8"))
      # self.each_sharp_img_3_pub.publish(self.bridge.cv2_to_imgmsg(each_sharp_img_3, "bgr8"))
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

  def get_binary(self, img, thr):
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
      ret, bin_img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY_INV)
      #ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

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

    x=45; y=60; w=40; h=40
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
      contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      out = np.zeros(img.shape,np.uint8)
      #print(len(contours))
      small_img = None
      if(len(contours) == 1):
        cnt = contours[0]
        [x,y,w,h] = cv2.boundingRect(cnt)
        
        roi_bin = img[y:y+h,x:x+w]
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
      return small_img

    except Exception as e:
      print(e)
      return None
    pass

  def detect_volume(self, img):
    """!
    @brief finally, we detect number using pre-train model
    @details

    @param[in] img: binary input image.
    
    @note input constraints: 
    @n - image should not be empty
    @n - image should be binary image (1 channel)

    @note output constraints: 
    @n - _img should not be empty 
    @n - morph_img should be gray image (1 channel)

    @return : output morphology image.
    """
    try:
      # input error check
      if img is None:
        raise NoImageError
      if (img.dtype != 'uint8'):
        raise MonoImageError(img.dtype)
      
      # doing
      contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      out = np.zeros(img.shape,np.uint8)
      #print(len(contours))
      for cnt in contours:
          [x,y,w,h] = cv2.boundingRect(cnt)
          # # compare with center point
          # M = cv2.moments(cnt, False)
          # cx = int(M['m10'] / M['m00'])
          # cy = int(M['m01'] / M['m00'])

          # in_center = self.judge_center(cx, cy, roi_w, roi_h)
          #print(cx)
          #print(in_center)

          # if(in_center == False):
          #   return -1, semantic_image

          # cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
          roi_bin = img[y:y+h,x:x+w]
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

class InputImageSizeError(Exception):
    def __init__(self, w, h):
      self.msg = "size of image is not 640x480. the size of image is " + str(w) + "x" + str(h) + "."
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
      self.msg = "the range should be 0~255. but the value is "+ str(c)
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
      self.msg = "the number contour should be 1. but the number of contour is " + str(c)
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



