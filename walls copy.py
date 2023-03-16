import cv2
import numpy as np
from utils import ray_from_camera, intersect_ray_plane, np_R_from_quat, unpack_vec

from typing import Tuple, List, Optional, Union, NamedTuple

from sensor_msgs.msg import PointCloud, Image
from std_msgs.msg import Header
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point

import tf2_ros
import tf2_geometry_msgs

import rospy
from tf.transformations import quaternion_matrix

from dataclasses import dataclass

from visualization_msgs.msg import Marker, MarkerArray
import time
# TF
from walls_alg import *

def contours_edges(mask, cnt_area_min, cnt_area_max):
    out = np.zeros(mask.shape, dtype="uint8")
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    i_s = [i for i, j in enumerate(contours) if areas[i] < cnt_area_max and areas[i] > cnt_area_min]
    contours_f = [contours[i] for i in i_s]

    cv2.drawContours(out, contours, -1, 255, 1)
    
    out[:, 0:10] = 0
    out[0:10, :] = 0
    # out[:, :] = 0
    


class WallDetector:
    mask = [
        np.array([0, 0, 124]),
        np.array([103, 78, 255])
    ]
    opening_kernel_size = 6
    erode_kernel_size = 6
    opening_iterations = 1
    debug = False
    resolution = 0.03
    map_wh = (7, 4)
    map_origin = np.array([0, 0])
    
    def __init__(self, cm: Optional[np.ndarray] = None, dc: Optional[np.ndarray] = None, tf_buffer = None, tf_listener = None, cv_bridge = None  ) -> None:
        self.cm = cm/2; self.cm[2, 2] = 1
        self.dc = dc 
        self.tf_listener = tf_listener
        self.tf_buffer = tf_buffer
        self.cv_bridge = cv_bridge
        if self.cv_bridge is None or self.tf_buffer is None:
            raise ValueError("self.cv_bridge is None")
        
        self.legs_mask = cv2.resize(cv2.imread("legs_mask.png", 0), (160, 120))

        self.point_cloud_pub = rospy.Publisher('/a/point_cloud', PointCloud, queue_size=10)
        self.debug_img_pub = rospy.Publisher("/a/debug_img", Image, queue_size=10)
        self.map_img_pub = rospy.Publisher("/a/map", Image, queue_size=10)
        self.markers_arr_pub = rospy.Publisher("/a/walls_viz", MarkerArray)

        map_a_wh = (int(self.map_wh[0]/self.resolution), int(self.map_wh[1]/self.resolution))
        self.map_a = np.zeros((map_a_wh[1], map_a_wh[0]), dtype=np.uint8)
        self.pls = Pls(dict(), [])
        self.wall_alg_params = Params(
            ldf=simple_distance, 
            assign_to_prev_th=3,
            dbscan_eps=1,
            connect_dist_th=0.3
        )

    def map_xy2ij(self, xy):
        ij = np.round((xy.reshape(-1)[:2] - self.map_origin) / self.resolution).astype(np.int)[::-1]
        if ij[0] < 0 or ij[1] < 0 or ij[0] >= self.map_a.shape[0] or ij[1] >= self.map_a.shape[1]:
            return None
        return ij


    def map_ij2xy(self, ij):
        xy = self.resolution*np.array(ij)[::-1] + self.map_origin
        return xy


    def set_cm_dc(self, cm: np.ndarray, dc: np.ndarray):
        self.cm = cm/2; self.cm[2, 2] = 1
        self.dc = dc 


    def on_pnts(self, pnts: List[np.ndarray]):
        debug_map = np.zeros((*self.map_a.shape, 3), dtype="uint8")

        xy = np.array(pnts)[:, :2]
        # print("xy:", xy)
        ijs = np.round((xy.reshape(-1, 2)[:, :2] - self.map_origin) / self.resolution).astype(np.int)[:, ::-1]
        ijs = ijs[(ijs[:, 0]>=0) & (ijs[:, 1]>=0)& (ijs[:, 1] < self.map_a.shape[1])& (ijs[:, 0] < self.map_a.shape[0])]
        # print("ijs:", ijs)
        local_map = np.zeros(self.map_a.shape, dtype=np.uint8)
        
        
        local_map[ijs[:, 0], ijs[:, 1]] = 255
        linesP = cv2.HoughLinesP(local_map, 1, np.pi / 180, 25, None, 10, 50)
        debug_map[:, :, 0] = local_map.copy()
        if linesP is not None:
            hls = [LineSP.from_2pnt(*[self.map_ij2xy([l[0][1], l[0][0]]), self.map_ij2xy([l[0][3], l[0][2]])]) for l in linesP]
            hls = [hl for hl in hls if abs(np.dot(hl.v, np.array([1, 0]))/np.dot(hl.v, hl.v)) < 0.1 or abs(np.dot(hl.v, np.array([0,1]))/np.dot(hl.v, hl.v)) < 0.1]
            # print("hls", hls)
            self.pls = proc_new(hls, self.pls, self.wall_alg_params)
            # print("pls", self.pls.pls)
            for i in range(0, len(hls)): 
                l = hls[i]
                # cv2.line(debug_map, tuple(self.map_xy2ij(l.p0).astype(np.int)[::-1]), tuple(self.map_xy2ij(l.p1).astype(np.int)[::-1]), (20,150,20), 2, cv2.LINE_AA)
        # for p in pnts:
        #     ij = self.map_xy2ij(p[:2])
        #     if ij is None:
        #         continue
        #     self.map_a[ij[0], ij[1]] = np.clip(self.map_a[ij[0], ij[1]] + 15, 0, 255)
            # self.map_a[ij[0], ij[1]] = 255
        self.map_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(np.flip(debug_map, 0), "bgr8")) # mono

    def proc_map(self):
        debug_map = np.zeros((*self.map_a.shape, 3), dtype="uint8")

        # eroded = cv2.erode(self.map_a, np.ones((5, 5),np.uint8), iterations=1)
        # opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, np.ones((3, 3),np.uint8), iterations=1)
        # dilate = cv2.dilate(opening, np.ones((5, 5),np.uint8), iterations=1)
        # if self.debug:
        #     cv2.imshow("dilate_map_a", dilate)

        ## hough
        # linesP = cv2.HoughLinesP(255*(self.map_a > 100).astype("uint8"), 1, np.pi / 180, 25, None, 10, 50)
        
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        #         cv2.line(debug_map, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
        ress = []
        if True:
            # hls = [LineSP.from_2pnt(*[self.map_ij2xy([l[0][1], l[0][0]]), self.map_ij2xy([l[0][3], l[0][2]])]) for l in linesP]
            # hls = [hl for hl in hls if abs(np.dot(hl.v, np.array([1, 0]))/np.dot(hl.v, hl.v)) < 0.1 or abs(np.dot(hl.v, np.array([0,1]))/np.dot(hl.v, hl.v)) < 0.1]
            # # print("hls", hls)
            # self.pls = proc_new(hls, Pls(dict(), []), self.wall_alg_params)
            # # print("pls", self.pls.pls)
            # for i in range(0, len(hls)): 
            #     l = hls[i]
            #     cv2.line(debug_map, tuple(self.map_xy2ij(l.p0).astype(np.int)[::-1]), tuple(self.map_xy2ij(l.p1).astype(np.int)[::-1]), (20,150,20), 2, cv2.LINE_AA)
            #     marker = Marker()
            #     marker.header.frame_id = "aruco_map"
            #     marker.header.stamp = rospy.Time.now()
            #     marker.ns = "color_markers"
            #     marker.id = i
            #     marker.type =  Marker.LINE_LIST
            #     marker.action = Marker.ADD
            #     marker.points = [Point(i[0], i[1], 0) for i in xys]
            #     marker.color.a = 0.5
            #     marker.pose.position.x = 0
            #     marker.pose.position.y = 0
            #     marker.pose.position.z = 0
            #     marker.pose.orientation.x = 0.0
            #     marker.pose.orientation.y = 0.0
            #     marker.pose.orientation.z = 0.0
            #     marker.pose.orientation.w = 1.0
            #     marker.color.r = 1
            #     marker.color.g = 0
            #     marker.color.b = 1
            #     marker.scale.x = 0.01
            #     ress.append(marker)

            for i in range(0, len(self.pls.pls)):
                marker = Marker()
                marker.header.frame_id = "aruco_map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "color_markers"
                marker.id = i
                marker.type =  Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.points = [Point(i[0], i[1], 0) for i in [self.pls.pls[i].p0, self.pls.pls[i].p1]]
                marker.color.a = 0.5
                marker.pose.position.x = 0
                marker.pose.position.y = 0
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.color.r = 0
                marker.color.g = 1
                marker.color.b = 0
                marker.scale.x = 0.05
                ress.append(marker)

        self.markers_arr_pub.publish(MarkerArray(markers=ress))
    
        # debug_map[:, :, 0] = self.map_a.copy()
        # self.map_a = dilate
        # self.map_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(np.flip(debug_map, 0), "bgr8")) # mono
        return None


    def on_frame(self, img: np.ndarray, mask_floor: np.ndarray, hsv: Optional[np.ndarray] = None) -> None:
        img_0 = cv2.resize(img, (160, 120))
        t1 = time.time()
        if hsv is None:
            hsv_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV)
        else:
            hsv_0 = cv2.resize(hsv, (160, 120))
        if self.cm is None:
            return None
        debug = img_0.copy()
        mask_floor_0 = cv2.resize(mask_floor, (160, 120))
        hsv_mask = cv2.inRange(hsv_0, self.mask[0], self.mask[1])
        
        mask = hsv_mask.copy()
        mask = cv2.erode(mask, np.ones((self.erode_kernel_size, self.erode_kernel_size),np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((self.opening_kernel_size, self.opening_kernel_size),np.uint8), iterations=self.opening_iterations)
        mask = cv2.dilate(mask, np.ones((self.erode_kernel_size, self.erode_kernel_size),np.uint8), iterations=1)

        canny = cv2.Canny(mask,100,200)      

        canny_masked = cv2.bitwise_and(canny, canny, mask=self.legs_mask)
        canny_masked = cv2.bitwise_and(canny_masked, canny_masked, mask=cv2.bitwise_not(mask_floor_0))
        
        points = np.array(np.where(canny_masked))[::-1, :].astype(np.float64).T
        print("mask: ", 1000*(time.time() - t1)); t1= time.time()
        for i in points:
            cv2.circle(debug,(int(i[0]),int(i[1])), 1, (0,0,255), -1)
        # undistort points
        if len(points) > 0:
            pnt_img_undist = cv2.undistortPoints(points.reshape(-1, 1, 2), self.cm, self.dc, None, None).reshape(-1, 2).T
            ray_v = np.ones((3, pnt_img_undist.shape[1]))
            ray_v[:2, :] = pnt_img_undist
            # ray_v = np.linalg.pinv(self.cm) @ ray_v
            ray_v /= np.linalg.norm(ray_v, axis=0)

            print(ray_v.shape)
            print("undistortPoints: ", 1000*(time.time() - t1)); t1= time.time()
        
            # if self.tf_buffer is not None:
                # print()
            try:
                transform = self.tf_buffer.lookup_transform("aruco_map", "main_camera_optical", rospy.Time())
            except tf2_ros.ConnectivityException:
                print("LookupException")
                return None
            except tf2_ros.LookupException:
                print("LookupException")
                return None
            R_wb = np.array(quaternion_matrix([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]))[:3, :3]
            t_wb = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
            # ray_v = np.array(
            #     [[0, 0, -1]]
            # ).T
            # print(R_wb)
            
            # ray_v = np.array([unpack_vec(tf2_geometry_msgs.do_transform_vector3(Vector3Stamped(vector=Vector3(v[0], v[1], v[2])), transform)) for v in ray_v.T])
            ray_v = np.matmul(R_wb, ray_v).T
            ray_o = t_wb

            pnts = [intersect_ray_plane(v, ray_o) for v in ray_v]
            pnts = [p for p in pnts if p is not None]
            # print(pnts)
            print("transform: ", 1000*(time.time() - t1)); t1= time.time()
            self.on_pnts(pnts)
            print("on_pnts: ", 1000*(time.time() - t1)); t1= time.time()
            self.point_cloud_pub.publish(PointCloud(header=Header(frame_id="aruco_map", stamp=rospy.Time.now()), points=[Point32(p[0], p[1], p[2]) for p in pnts]))
        
        self.proc_map()

        print("proc_map: ", 1000*(time.time() - t1)); t1= time.time()
        
        self.debug_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug, "bgr8"))
        if self.debug:
            cv2.imshow("debug", debug)
            cv2.imshow("hsv_mask", hsv_mask)
            cv2.imshow("mask", mask)
            cv2.imshow("canny_masked", canny_masked)
            cv2.imshow("legs_mask", self.legs_mask)
            cv2.imshow("map_a", np.flip(self.map_a, 0))
        

