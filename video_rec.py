#!/usr/bin/env python3

import rospy

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import time

class Recorder:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/main_camera/image_raw", Image, self.callback)

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.size = (320, 240)
        self.video = cv2.VideoWriter(f'output_{time.time()}.avi', fourcc, 20.0, self.size)

    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        image = cv2.resize(image, self.size)
        self.video.write(image)

    def close(self):
        self.video.release()

def main():
    rec = Recorder()
    rospy.init_node('video_recorder', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Exit')
    finally:
        rec.close()
