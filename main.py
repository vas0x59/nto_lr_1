#!/usr/bin/env python3
import rospy

import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import time
import walls
import fire
from utils import *

import tf2_ros

import math
from clover import srv
from std_srvs.srv import Trigger

import numpy as np

import threading

class NodeHandler:
    floor_thr = [
                np.array([0, 0, 0]),
        np.array([180, 255, 120])
    ]

    def __init__(self, video_file=None):
        self.is_start = False

        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        self.fires = fire.FireSearch()
        
        if video_file is not None:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap:
                raise Exception(f"Couldn`t open video file {video_file}")

            self.cm = np.array([[ 92.37552066, 0., 160.5], [0., 92.37552066, 120.5], [0., 0., 1.]], dtype="float64")
            self.dc = np.zeros(5, dtype="float64")

            rospy.Timer(rospy.Duration(1/30), self.video_callback)
        else:
            self.cm, self.dc = camera_cfg_cvt(
                    rospy.wait_for_message("/main_camera/camera_info", CameraInfo))
        self.wd = walls.WallDetector(cm=self.cm, dc=self.dc, tf_buffer=self.tfBuffer, tf_listener=self.listener, cv_bridge=self.bridge)
        self.fires = fire.FireSearch(cm=self.cm, dc=self.dc, tf_buffer=self.tfBuffer, cv_bridge=self.bridge)
        if video_file is None:
            self.floor_mask_pub = rospy.Publisher("/a/floor_mask", Image)
            self.image_sub = rospy.Subscriber("/main_camera/image_raw_throttled", Image, self.callback)
        self.th = threading.Thread(target=self.thr_target, daemon=True)
        self.stop = False
        self.th.start()
    
    def thr_target(self):
        while True:
            if self.stop:
                break
            self.wd.proc_map()
            time.sleep(1)

    def enable(self):
        self.is_start = True

    def disable(self):
        self.is_start = False
        
    def floor_mask(self, hsv):
        hsv = cv2.blur(hsv, (10, 10))
        mask = cv2.inRange(hsv, self.floor_thr[0], self.floor_thr[1])
        #mask = cv2.bitwise_not(mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        mask = np.zeros(mask.shape, dtype="uint8")
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            
            area = cv2.contourArea(approx)
            if area < 600:
                continue
        
            mask = cv2.fillPoly(mask, pts = [approx], color=(255,255,255))
        
        return mask

    def video_callback(self, event):
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("End of video file")

        self.callback(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if not self.is_start:
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        floor_mask = self.floor_mask(hsv)
        self.floor_mask_pub.publish(self.bridge.cv2_to_imgmsg(floor_mask, "mono8"))

        self.wd.on_frame(img=image, mask_floor=floor_mask, hsv=hsv)

        #fire_border = 20
        #h, w, _ = image.shape

        #x1, y1, x2, y2 = fire_border - 1, fire_border - 1, w - fire_border + 1, h - fire_border + 1
        #self.fires.on_frame(image[y1:y2, x1:x2], 
        #        mask_floor=floor_mask[y1:y2, x1:x2], 
        #        hsv=hsv[y1:y2, x1:x2])
        self.fires.on_frame(image, mask_floor=floor_mask, hsv=hsv)

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
    
    handler = NodeHandler(video_file=None)
    
    try:
        FLIGHT_HEIGHT = 1

        FLY_BY_SQUARE = False

        clover_path = [0, 1, 0, 4, 7, 4, 7, 0, 1, 0]
        if not FLY_BY_SQUARE:
            clover_path = [0, 2, 0, 3, 1, 3.5, 1.5, 3, 2, 4, 4, 4, 4, 0.9, 4, 2.5, 6.5, 2.5, 6.7, 0.5, 6.5, 3.5, 4.5, 3, 4.5, 4, 2, 4, 1.5, 3, 0, 3, 0, 2]
            
  
        navigate_wait(z=FLIGHT_HEIGHT, speed = 0.6, frame_id='body', auto_arm=True)
        rospy.sleep(2)
        print('takeoff')

        t = get_telemetry(frame_id='aruco_map')
        land_position = (t.x, t.y)
        rospy.sleep(1)

        for i in range(0, len(clover_path), 2):
            if i == 4:
                handler.enable()

            navigate_wait(x=clover_path[i], y=clover_path[i+1], z=FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
            print("go to next point...")


        handler.disable()

        navigate_wait(x=land_position[0], y=land_position[1], z=FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
        rospy.sleep(3)
        land()

        handler.fires.report()
        handler.wd.report()

        handler.stop = True
        exit(0)

        rospy.spin()

    except KeyboardInterrupt:
        handler.fires.report()
        handler.wd.report()
        
        handler.stop = True

if __name__ == "__main__":
    main()
