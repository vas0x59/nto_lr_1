#!/usr/bin/env python3
import rospy

import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import time

import walls 
from utils import camera_cfg_cvt

import tf2_ros
import numpy as np

class Node:
    def __init__(self):
        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.wd = walls.WallDetector(tf_buffer=self.tfBuffer, tf_listener=self.listener, cv_bridge=self.bridge)
        self.cm = None
        self.dc = None
        self.image_sub = rospy.Subscriber("/main_camera/image_raw", Image, self.callback)
        self.data_sub = rospy.Subscriber("/main_camera/camera_info", CameraInfo, self.camera_info_clb)

    def camera_info_clb(self, msg):
        print("cminfo")
        self.cm, self.dc = camera_cfg_cvt(msg)
        self.wd.set_cm_dc(self.cm, self.dc)
        self.data_sub.unregister()
        
    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("clb")
            self.wd.on_frame(img=image, mask_floor=np.ones(image.shape[:2], dtype="uint8"))
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('video_recorder', anonymous=True)
    rec = Node()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Exit')

if __name__ == "__main__":
    main()