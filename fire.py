import cv2
import numpy as np

from utils import *

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point
import rospy

from visualization_msgs.msg import Marker, MarkerArray

class FireSearch:
    lower_thr = (
        (0, 30, 30),
        (170, 30, 30)
    )
    upper_thr = (
        (35, 255, 255),
        (180, 255, 255)
    )

    fire_fraction = 0.0005
    fire_radius = 1

    def __init__(self, cm: Optional[np.ndarray] = None, dc: Optional[np.ndarray] = None, tf_buffer = None):
        self.cm = cm
        self.dc = dc
        self.tf_buffer = tf_buffer
        self.fires_pub = rospy.Publisher("/a/fires_viz", MarkerArray)
        self.fires = []
        
    def report(self):
        print(f"Fires: {len(self.fires)}")
        print()
        for idx, cloud in enumerate(self.fires):
            x, y = 0, 0
            for p in cloud:
                x += p[0]
                y += p[1]
            x /= len(cloud)
            y /= len(cloud)
            print(f"Fire {idx + 1}: {round(x, 1)} {round(y, 1)}")

    def mask_overlay(self, frame):
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        for lower, upper in zip(self.lower_thr, self.upper_thr):
            mask = cv2.bitwise_or(cv2.inRange(frame, lower, upper), mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        return mask

    def find_closest(self, point):
        distances = []
        for fire in self.fires:
            distances.append((fire[0][0] - point[0]) ** 2 + (fire[0][1] - point[1]) ** 2)
        
        min_dist = min(distances)
        return distances.index(min_dist), min_dist

    def insert_fire(self, point):
        if len(self.fires) == 0:
            self.fires.append([point])
            return

        idx, distance = self.find_closest(point)
        if distance <= self.fire_radius:
            self.fires[idx].append(point)
            return
        self.fires.append([point])

    def publish_markers(self):
        result = []
        iddd = 0
        for fs in self.fires:
            m = np.mean(fs, axis=0)
            # result.append(transform_marker(marker, frame_to="aruco_map"))
            # cx_map, cy_map, cz_map, _ = transform_xyz_yaw(
            # marker.cx_cam, marker.cy_cam, marker.cz_cam, 0, "main_camera_optical", frame_to, listener)
            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = iddd
            marker.type =  Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1

            marker.color.a = 0.8

            marker.color.r = 1
            marker.color.g = 0.1
            marker.color.b = 0

            result.append(marker)
            iddd += 1
        self.fires_pub.publish(MarkerArray(markers=result))
        return None

    def on_frame(self, frame, mask_floor):
        hsv = cv2.cvtColor(
                frame, 
                cv2.COLOR_BGR2HSV)

        mask = self.mask_overlay(hsv)
        
        mask = cv2.bitwise_and(mask, mask_floor)

        contours = cv2.findContours(mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)[-2]

        frame_vol = np.prod(frame.shape[0:2])

        assert frame_vol != 0
        contours = list(filter(
                lambda c: (cv2.contourArea(c) / frame_vol) >= self.fire_fraction, 
                contours))

        if len(contours) > 0:
            pnt_img = []
            for cnt in contours:
                _, _, w, h = cv2.boundingRect(cnt)
                ratio = w / h
                
                if abs(1.0 - ratio) >= 0.4:
                    continue
                
                M = cv2.moments(cnt)

                pnt_img.append(
                    [int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"])])
                #self.insert_fire(pnt_img[-1])
                #cv2.circle(frame, pnt_img[-1], 6, (255, 0, 0), 2)

            pnt_img = np.array(pnt_img).astype(np.float64)
            pnt_img_undist = cv2.undistortPoints(pnt_img.reshape(-1, 1, 2), self.cm, self.dc, None, None).reshape(-1, 2).T
            ray_v = np.ones((3, pnt_img_undist.shape[1]))
            ray_v[:2, :] = pnt_img_undist
            ray_v /= np.linalg.norm(ray_v, axis=0)

            print(ray_v.shape)
            if self.tf_buffer is not None:
                try:
                    transform = self.tf_buffer.lookup_transform("aruco_map", "main_camera_optical", rospy.Time())
                except tf2_ros.LookupException:
                    print("tf2_ros.LookupException")
                    return None
                # R_wb = np.array(quaternion_matrix([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))[:3, :3]
                t_wb = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
                # ray_v = np.array(
                #     [[0, 0, -1]]
                # ).T
                # print(R_wb)
                ray_v = np.array([unpack_vec(tf2_geometry_msgs.do_transform_vector3(Vector3Stamped(vector=Vector3(v[0], v[1], v[2])), transform)) for v in ray_v.T])
                ray_o = t_wb

                pnts = [intersect_ray_plane(v, ray_o) for v in ray_v]
                pnts = [p for p in pnts if p is not None]
                print(pnts)
                for p in pnts:
                    self.insert_fire(p[:2])
        self.publish_markers()
        # cv2.imshow('frame', frame)
        # cv2.imshow('mask_floor', mask_floor)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(10)
