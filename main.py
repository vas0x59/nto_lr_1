#!/usr/bin/env python3
import rospy

import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import walls
import fire
from utils import *

import tf2_ros

import math
from clover import srv
from std_srvs.srv import Trigger

import numpy as np

class NodeHandler:
    floor_thr = [
        np.array([0, 0, 124]),
        np.array([103, 78, 255])
    ]

    def __init__(self, video_file=None):
        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        self.fires = fire.FireSearch()
        
        if video_file is not None:
            self.cap = cv2.VideoCapture(video_file)
            return

        self.image_sub = rospy.Subscriber("/main_camera/image_raw", Image, self.callback)
        self.cm, self.dc = camera_cfg_cvt(
                rospy.wait_for_message("/main_camera/camera_info", CameraInfo))
        self.wd = walls.WallDetector(cm=self.cm, dc=self.dc, tf_buffer=self.tfBuffer, tf_listener=self.listener, cv_bridge=self.bridge)
        self.fires = fire.FireSearch(cm=self.cm, dc=self.dc, tf_buffer=self.tfBuffer)
        
    def floor_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.floor_thr[0], self.floor_thr[1])
        mask = cv2.bitwise_not(mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        mask = np.zeros(mask.shape, dtype="uint8")
        for cnt in contours:
            #hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
            
            area = cv2.contourArea(approx)
            if area < 6000:
                continue
        
            mask = cv2.fillPoly(mask, pts = [approx], color=(255,255,255))
        return mask

    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.wd.on_frame(img=image, mask_floor=self.floor_mask(image))
        self.fires.on_frame(image, mask_floor=self.floor_mask(image))
        #cv2.imshow('floor_mask', self.floor_mask(image))
        #cv2.imshow('frame', image)
        #cv2.waitKey(1)

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.5, frame_id='', auto_arm=False, tolerance=0.15):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

def main():
    rospy.init_node('first_task', anonymous=True)
    
    #VIDEO_FILE = 'output.avi'
    VIDEO_FILE = None
    handler = NodeHandler()
    
    try:  
        if VIDEO_FILE is not None:
            while not rospy.is_shutdown():
                ret, frame = handler.cap.read()
                
                if not ret:
                    break
                handler.fires.on_frame(frame, mask_floor=handler.floor_mask(frame))
        
            handler.fires.report()
        
            return
        
        FLIGHT_HEIGHT = 1
                
        navigate_wait(z=FLIGHT_HEIGHT, speed = 0.6, frame_id='body', auto_arm=True)
        rospy.sleep(2)
        print('takeoff')

        telem = get_telemetry(frame_id='aruco_map')
        xf = telem.x
        yf = telem.y
        rospy.sleep(1)

        s = [0, 1, 0, 4, 7, 4, 7, 0, 1, 0]

        for i in range(0, len(s), 2):
            navigate_wait(x=s[i], y=s[i+1], z=FLIGHT_HEIGHT, speed=0.25, frame_id="aruco_map")
            rospy.sleep(2)
            
            print("go to next point...")

        handler.fires.report()

        navigate_wait(x=xf, y=yf, z=FLIGHT_HEIGHT, speed=0.25, frame_id="aruco_map")
        rospy.sleep(3)
        land()

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
