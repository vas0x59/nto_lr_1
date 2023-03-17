#!/usr/bin/env python3
import rospy

import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
import time
import walls
from walls_alg import WallStates
import fire
from utils import *

import tf2_ros

import math
from clover import srv
from std_srvs.srv import Trigger

import numpy as np
from enum import Enum
import threading

USE_GPIO = False

gpio_obj = None
if USE_GPIO:
    import pigpio

    class GPIOHandler:
        TYPE_A_PIN = 22
        TYPE_B_PIN = 27

        def __init__(self):
            self.pi = pigpio.pi()
            self.pi.set_mode(self.TYPE_A_PIN, pigpio.OUTPUT)
            self.pi.set_mode(self.TYPE_B_PIN, pigpio.OUTPUT)

            self.clear()

        def clear(self):
            self.pi.write(self.TYPE_A_PIN, 0)
            self.pi.write(self.TYPE_B_PIN, 0)

        def push_object(self, type_o):
            a_pin = (1 if type_o == 'A' else 0)
            b_pin = (not a_pin) 

            self.pi.write(self.TYPE_A_PIN, a_pin)
            self.pi.write(self.TYPE_B_PIN, b_pin)
            time.sleep(0.005)
            self.clear()
    gpio_obj = GPIOHandler()


FLIGHT_HEIGHT = 1.2

FLY_BY_SQUARE = False

clover_path = [0, 1, 0, 4, 7, 4, 7, 0, 1, 0]
if not FLY_BY_SQUARE:
    """
    clover_path = [\
        0, 3, \
        1, 3, \
        2, 3, \
        2, 4, \
        4, 4, \
        4, 2, \
        4, 4, \
        5, 4, \
        6, 4, \
        6, 1.5, \
        6, 4]
    """
    clover_path = [0, 3, 1, 3, 2, 3]
    
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

class SMStates(Enum):
    Start = 0
    Gen = 1
    Going = 2
    CheckForward = 3
    Oblet = 4
    WithOutOblet = 5
    End = 6
    Fire = 7


class NodeHandler:
    # Пороговые значения для пола
    floor_thr = [
        np.array([0, 0, 0]),
        np.array([180, 255, 120])
    ]
    clover_size = 0.42
    pnts_sep = 0.15

    def __init__(self, video_file=None):
        self.is_start = False

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.fires = fire.FireSearch()

        self.follow_wli = 0
        self.point_id = 0
        self.points = np.array([])
        self.update_t = 0

        self.state = SMStates.Start
        
        if video_file is not None:
            # Для тестирования кода (с использованием видео)
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap:
                raise Exception(f"Couldn`t open video file {video_file}")

            self.cm = np.array([[ 92.37552066, 0., 160.5], [0., 92.37552066, 120.5], [0., 0., 1.]], dtype="float64")
            self.dc = np.zeros(5, dtype="float64")

            rospy.Timer(rospy.Duration(1/30), self.video_callback)
        else:
            self.cm, self.dc = camera_cfg_cvt(
            rospy.wait_for_message("/main_camera/camera_info", CameraInfo))

        self.wd = walls.WallDetector(cm=self.cm, dc=self.dc, tf_buffer=self.tf_buffer, tf_listener=self.listener, cv_bridge=self.bridge, start_point=np.array([0, 0]))
        self.fires = fire.FireSearch(cm=self.cm, dc=self.dc, tf_buffer=self.tf_buffer, cv_bridge=self.bridge, on_fire=self.on_fire)
        if video_file is None:
            self.floor_mask_pub = rospy.Publisher("/a/floor_mask", Image)
            self.image_sub = rospy.Subscriber("/main_camera/image_raw_throttled", Image, self.callback)
        
        self.path_pub = rospy.Publisher("/a/path_points", MarkerArray)

        self.th = threading.Thread(target=self.thr_target, daemon=True)
        self.stop = False
        self.th.start()
        self.fire_point = None
        self.fire_type = None

    def on_fire(self, fire_point, fire_type):
        self.fire_state = self.state
        self.state = SMStates.Fire
        self.fire_point = np.array(fire_point)[:2]
        self.fire_type = fire_type


    def generate_next_points(self, line: LineSP, current_point: np.ndarray = None, offset_end: float = None, offset_start: float = None):
        if offset_end is None:
            offset_end = self.clover_size
        if offset_start is None:
            offset_start = self.clover_size
        if current_point is not None:
            t0 = np.clip(param_of_nearest_pnt_on_line(current_point, line), 0, line.t_max)
        else:
            t0 = 0
        if ((line.t_max - offset_end) - t0) < (self.pnts_sep + 0.001):
            return []
        ar = np.linspace(np.clip(t0 + offset_start, 0, line.t_max), np.clip(line.t_max - offset_end, 0, line.t_max), int(np.clip(line.t_max / self.pnts_sep, 0, 10000)))
        points = []

        lnorm = line.left_norm()
        for t in ar:
            line_ = LineSP(line.p_t(t), lnorm, self.clover_size)
            points.append(line_.p1)   
        return points
    def gen_next_point(self, line: LineSP, current_point: np.ndarray, offset_end: float = None):
        if offset_end is None:
            offset_end = self.clover_size
        t0 = np.clip(param_of_nearest_pnt_on_line(current_point, line), 0, line.t_max)
        t = np.clip(t0 + self.pnts_sep, 0, line.t_max - offset_end)
        lnorm = line.left_norm()
        line_ = LineSP(line.p_t(t), lnorm, self.clover_size)
        return line_.p1

    def get_body_t(self):
        try:
            transform = self.tf_buffer.lookup_transform("aruco_map", "body", rospy.Time())
        except tf2_ros.ConnectivityException:
            print("LookupException")
            return None
        except tf2_ros.LookupException:
            print("LookupException")
            return None
        t_wb = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        return t_wb

    # Метод, обновляющий карту каждую 1 секунду
    def thr_target(self):
        while True:
            if self.stop:
                break
            self.wd.proc_map()
            
            self.update_t = time.time()
            print("state: ", self.state)
            

            
            
            t_wb = self.get_body_t()
            if t_wb is None:
                time.sleep(1)
                continue

            if (self.state == SMStates.Gen):
                if (self.follow_wli == 10):
                    self.state = SMStates.End
                if (self.follow_wli + 1)  <= len(self.wd.wp.state_obj.walls):
                    self.points = self.generate_next_points(self.wd.wp.state_obj.walls[self.follow_wli].line)
                    self.point_id = 0
                    self.state = SMStates.Going
                    marker_array = []
                    for i, point in enumerate(self.points):
                        marker = self.point_marker(i, point)
                        marker_array.append(marker)

                    self.path_pub.publish(MarkerArray(markers=marker_array))
            
            # if (self.point_id + 1) <= len(self.points):
            #     pt = self.points[self.point_id]
            #     navigate_wait(x=pt[0], y=pt[1], z=FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
            #     self.point_id += 1

            # walls = [wall for wall in self.wd.wp.state_obj.walls if wall.state == States.FIXED]
            # if (self.follow_wli + 1) > len(walls):
            #     continue

            # self.point_id = 0
            # self.points = self.gen_point(walls[self.follow_wli])

            # marker_array = []
            # for i, point in enumerate(self.points):
            #     marker = self.point_marker(i, point)
            #     marker_array.append(marker)

            # self.path_pub.publish(MarkerArray(markers=marker_array))

            # self.follow_wli += 1
            time.sleep(1)

            

    def point_marker(self, i, pose):
        marker = Marker()
        marker.header.frame_id = "aruco_map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "color_markers"
        marker.id = i
        marker.type =  Marker.SPHERE
        marker.action = Marker.ADD

        # Позиция и ориентация
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Масштаб
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Цвет
        marker.color.a = 0.8

        marker.color.r = 0
        marker.color.g = 1.0
        marker.color.b = 0.1

        return marker

    def gen_point(self, wall):
        prop = 0.05
        clover_size = 0.25

        max_t = wall.line.t_max.item()
        lnorm = wall.line.left_norm()
        
        points = []
        for t in np.arange(clover_size, max_t + clover_size, max_t * prop):
            line = LineSP(wall.line.p_t(t), lnorm, clover_size)
            points.append(line.p1)           

        return points

    def enable(self):
        self.is_start = True
        if (self.state == SMStates.Start):
            self.state = SMStates.Gen

    def disable(self):
        self.is_start = False
        
    # Метод, создающий маску для полаы
    def floor_mask(self, hsv):
        hsv = cv2.blur(hsv, (10, 10))
        mask = cv2.inRange(hsv, self.floor_thr[0], self.floor_thr[1])

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

    # Метод, реализующий публикацию видеопотока (используется для тестирование кода)
    def video_callback(self, event):
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("End of video file")

        self.callback(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    # Callback-метод топика с изображением
    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if not self.is_start:
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Создаем маску для пола площадки
        floor_mask = self.floor_mask(hsv)
        self.floor_mask_pub.publish(self.bridge.cv2_to_imgmsg(floor_mask, "mono8"))

        # Обрабатываем новый кадр из топика
        self.wd.on_frame(img=image, mask_floor=floor_mask, hsv=hsv)
        self.fires.on_frame(image, mask_floor=floor_mask, hsv=hsv)

    def sbros(self, t):
        print("sb type: ", t)
        if USE_GPIO:
            gpio_obj.push_object(t)

MAGIC = False

def main():
    rospy.init_node('first_task', anonymous=True)
    
    handler = NodeHandler(video_file=None)
    
    try:
        

        # Пролетаем по координатам
  
        navigate_wait(z=FLIGHT_HEIGHT, speed = 0.6, frame_id='body', auto_arm=True)
        rospy.sleep(2)
        print('takeoff')

        t = get_telemetry(frame_id='aruco_map')
        land_position = (t.x, t.y)
        rospy.sleep(1)

        for i in range(0, len(clover_path), 2):
            if i == 4:
                # При подлете к рабочей зоне, включаем распознавание пожаров, стен и пострадавших
                handler.enable()

            navigate_wait(x=clover_path[i], y=clover_path[i+1], z=FLIGHT_HEIGHT, speed=0.5, frame_id="aruco_map")
            print("go to next point...")


        CheckForward_states = 0
        CheckForward_1_t_started = 0
        CheckForward_pathblocked = 0
        CheckForward_checks = 0

        navs_pnts = []
        if not MAGIC:
            while True:
                print("state: ", handler.state)
                t_wb = handler.get_body_t()
                if t_wb is None:
                    time.sleep(1)
                    continue
                    
                if (handler.state == SMStates.Going):
                    print("Going: ", handler.point_id )
                    if (handler.wd.wp.state_obj.walls[handler.follow_wli].state == WallStates.UPDATABLE):
                        lll = handler.wd.wp.state_obj.walls[handler.follow_wli].line
                        # handler.points = handler.generate_next_points(handler.wd.wp.state_obj.walls[handler.follow_wli].line, t_wb[:2], handler.clover_size)
                        handler.points = handler.generate_next_points(handler.wd.wp.state_obj.walls[handler.follow_wli].line, t_wb[:2])
                        handler.point_id = 0
                        pt = handler.gen_next_point(lll, t_wb[:2])
                        handler.points =[pt]
                        navigate_wait(x=pt[0], y=pt[1], z=FLIGHT_HEIGHT, speed=0.4, frame_id="aruco_map")
                        navs_pnts.append(pt)
                        if abs(param_of_nearest_pnt_on_line(t_wb[:2], lll) - (lll.t_max - handler.clover_size) ) < 0.1:
                            handler.state = SMStates.CheckForward
                    else: 
                        if (handler.point_id + 1) <= len(handler.points):
                            pt = handler.points[handler.point_id]
                            navigate_wait(x=pt[0], y=pt[1], z=FLIGHT_HEIGHT, speed=0.4, frame_id="aruco_map")
                            navs_pnts.append(pt)
                            handler.point_id += 1
                        else:
                            handler.state = SMStates.CheckForward
                elif (handler.state == SMStates.CheckForward):
                    print("CheckForward_states", CheckForward_states)
                    print("CheckForward_checks", CheckForward_checks)
                    print("CheckForward_pathblocked", CheckForward_pathblocked)
                    if (CheckForward_states == 100):
                        CheckForward_states = 0
                    if (CheckForward_states == 0) and (time.time() - handler.update_t) < 0.8:
                        CheckForward_states = 1
                        CheckForward_checks = 0
                        CheckForward_pathblocked = 0
                        CheckForward_1_t_started = time.time()
                    if CheckForward_states == 1:
                        if (handler.follow_wli + 1) == len(handler.wd.wp.state_obj.walls):
                            pass
                        elif len(handler.wd.wp.state_obj.walls) > (handler.follow_wli + 1):
                            line_to_an =  handler.wd.wp.state_obj.walls[handler.follow_wli + 1].line
                            crt_line =  handler.wd.wp.state_obj.walls[handler.follow_wli].line
                            if (distance_to_line(t_wb[:2], line_to_an) < 5 * handler.clover_size) and np.dot(crt_line.left_norm(), line_to_an.v) > 0.8:
                                CheckForward_pathblocked += 1
                            else:
                                pass
                        CheckForward_checks += 1

                        if (time.time() - CheckForward_1_t_started) > 1 and (CheckForward_checks > 2):
                            CheckForward_states = 2 
                    if CheckForward_states == 2:
                        if CheckForward_pathblocked > 0:
                            handler.state = SMStates.WithOutOblet
                            CheckForward_states = 100
                        else:
                            handler.state = SMStates.Oblet
                            CheckForward_states = 100
                if (handler.state == SMStates.WithOutOblet):
                    if len(handler.wd.wp.state_obj.walls) > (handler.follow_wli + 1):
                        handler.follow_wli += 1
                        handler.state = SMStates.Gen
                if (handler.state == SMStates.Oblet):
                    line = handler.wd.wp.state_obj.walls[handler.follow_wli].line
                    lnorm = line.left_norm()
                    line_ = LineSP(line.p_t(handler.clover_size*1 + line.t_max), lnorm, handler.clover_size)
                    pt = line_.p1
                    navigate_wait(x=pt[0], y=pt[1], z=FLIGHT_HEIGHT, speed=0.5, frame_id="aruco_map")
                    navs_pnts.append(pt)
                    pt = line.p_t(handler.clover_size*1 + line.t_max)
                    navigate_wait(x=pt[0], y=pt[1], z=FLIGHT_HEIGHT, speed=0.5, frame_id="aruco_map")
                    navs_pnts.append(pt)
                    handler.follow_wli += 1
                    handler.state = SMStates.Gen
                    
                
                if (handler.state == SMStates.End):
                    break
                    
                if (handler.state == SMStates.Fire):
                    if len(navs_pnts) == 0:
                        last_pt = t_wb[:2]
                    else:
                        last_pt = navs_pnts[-1]
                    pt = handler.fire_point
                    navigate_wait(x=pt[0], y=pt[1], z=FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map", tolerance=0.3)
                    ### DO some staff
                    handler.sbros(handler.fire_type)
                    time.sleep(1)

                    navigate_wait(x=last_pt[0], y=last_pt[1], z=FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map", tolerance=0.2)
                    handler.state = handler.fire_state 
                    
                time.sleep(0.1)

            for pt in navs_pnts[::-1]:
                navigate_wait(x=pt[0], y=pt[1], z=FLIGHT_HEIGHT, speed=0.5, frame_id="aruco_map")
        else:
            xs = [0, 1, 2, 3, 4, 5, 6, 7]
            for x in xs:
                navigate_wait(x=x, y=4, z=FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
            for x in xs.reverse():
                navigate_wait(x=x, y=4, z=FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
        # Выключаем распознавание
        handler.disable()

        navigate_wait(x=land_position[0], y=land_position[1], z=FLIGHT_HEIGHT, speed=0.5, frame_id="aruco_map")
        rospy.sleep(3)
        land()

        # Отчет
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
