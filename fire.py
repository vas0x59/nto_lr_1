import cv2
import numpy as np

from utils import *

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point
import rospy

from visualization_msgs.msg import Marker, MarkerArray
import requests

class FireSearch:
    # Пороговые значения HSV для определения пожаров и пострадавших соответственно
    lower_thr = (
        (0, 30, 80),
        (170, 30, 80)
    )
    upper_thr = (
        (35, 255, 255),
        (180, 255, 255)
    )
    blue_lower = (
        95, 100, 50
    )
    blue_upper = (
        125, 255, 255
    )

    # Параметры определения пожаров
    fire_fraction = 0.0035
    fire_radius = 1

    def __init__(self, cm: Optional[np.ndarray] = None, dc: Optional[np.ndarray] = None, tf_buffer = None, cv_bridge = None, on_fire=None):
        # Параметры камеры для функции undistort
        self.cm = cm
        self.dc = dc
        self.on_fire = on_fire

        # TF буффер и cv_bridge для сообщений типа Image
        self.tf_buffer = tf_buffer
        self.cv_bridge = cv_bridge

        # Массив для хранения координат найденных пожаров и пострадавших
        self.fires = []
        self.blue_obj = []
        

        # Топики для rviz и отладки программы
        self.debug_pub = rospy.Publisher("/a/fires_debug", Image)
        self.mask_overlay_pub = rospy.Publisher("/a/fires_mask_overlay_pub", Image)
        self.fires_pub = rospy.Publisher("/a/fires_viz", MarkerArray)
        self.blue_pub = rospy.Publisher("/a/blue_viz", MarkerArray)

    # Метод определяющий материал пожара по координатам 
    def get_material(self, position):
        payload = {'x': position[0], 'y': position[1]}
        material = requests.get('http://65.108.222.51/check_material', params=payload).text

        return material
    def get_class(self, p):
        a_classes = ['coal', 'textiles', 'plastics']
        material = self.get_material(p)
        class_fire = ('A' if material in a_classes else 'B')
        return class_fire
    # Метод, формирующий отчет о найденных объектах
    def report(self):
        
        print(f"Fires: {len(self.fires)}")
        print()

        a_classes = ['coal', 'textiles', 'plastics']
        for idx, cloud in enumerate(self.fires):
            x, y = np.mean(cloud, axis=0)

            material = self.get_material([x, y])
            class_fire = ('A' if material in a_classes else 'B')
            print(f"Fire {idx + 1}: {round(x, 2)} {round(y, 2)} {material} {class_fire}")
        print()

        print(f"Injured: {len(self.blue_obj)}")
        print()

        for idx, cloud in enumerate(self.blue_obj):
            x, y = np.mean(cloud, axis=0)

            idd, dis = self.find_closest([x, y], self.fires)
            print(f"Injured {idx + 1}: {round(x, 2)} {round(y, 2)} {idd + 1}")

    # Метод создает маску по заданным пороговым значениям (для определения пожаров)
    def mask_overlay(self, frame):
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        for lower, upper in zip(self.lower_thr, self.upper_thr):
            mask = cv2.bitwise_or(cv2.inRange(frame, lower, upper), mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        return mask

    # Метод создает маску по заданным пороговым значениям (для определения пострадавших)
    def blue_overlay(self, frame):
        mask = cv2.inRange(frame, self.blue_lower, self.blue_upper)
        
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        return mask

    # 
    def find_closest(self, point, tuple_obj):
        distances = []
        for fire in tuple_obj:
            distances.append((fire[0][0] - point[0]) ** 2 + (fire[0][1] - point[1]) ** 2)
        
        min_dist = min(distances)
        return distances.index(min_dist), min_dist

    def insert_fire(self, point, idx):
        obj = (self.fires if idx == 0 else self.blue_obj)
        
        if len(obj) == 0:
            obj.append([point])
            self.on_fire(np.array(point), self.get_class(point))
            return

        idx, distance = self.find_closest(point, obj)
        if distance <= self.fire_radius:
            obj[idx].append(point)
            return
        obj.append([point])
        self.on_fire(np.array(point), self.get_class(point))

    # Метод, публикующий маркеры пожаров в rviz
    def publish_markers(self):
        result = []
        iddd = 0
        for fs in self.fires:
            # На основе множества распознаваний одного пострадавшего формируем усредненные координаты
            m = np.mean(fs, axis=0)

            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = iddd
            marker.type =  Marker.CUBE
            marker.action = Marker.ADD

            # Позиция и ориентация
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Масштаб
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1

            # Цвет
            marker.color.a = 0.8

            marker.color.r = 1
            marker.color.g = 0.1
            marker.color.b = 0

            result.append(marker)
            iddd += 1

        # Публикуем маркеры
        self.fires_pub.publish(MarkerArray(markers=result))
        return None

    # Метод, публикующий маркеры пострадавших в rviz
    def publish_markers_blue(self):
        result = []
        iddd = 0
        for fs in self.blue_obj:
            # На основе множества распознаваний одного пострадавшего формируем усредненные координаты
            m = np.mean(fs, axis=0)

            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = iddd
            marker.type =  Marker.CUBE
            marker.action = Marker.ADD

            # Позиция и ориентация
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Масштаб
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1

            # Цвет
            marker.color.a = 0.8

            marker.color.r = 0
            marker.color.g = 0.1
            marker.color.b = 1

            result.append(marker)
            iddd += 1

        # Публикуем маркеры
        self.blue_pub.publish(MarkerArray(markers=result))
        return None

    # Вспомогательный метод для определения расстояния между 2 точками
    def distance(self, a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def on_frame(self, frame, mask_floor, hsv: Optional[np.ndarray] = None):
        if hsv is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Создаем маски для нахождения пожаров и пострадавших
        debug = frame.copy()
        mask_overlay = self.mask_overlay(hsv)

        mask_blue = self.blue_overlay(hsv)
        
        # Создаем маску для пола площадки
        contours_floor = cv2.findContours(mask_floor, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)[-2]

        cnt_floor = sorted(contours_floor , key=cv2.contourArea)[-1]

        mannualy_contour = []

        convex_floor = cv2.convexHull(cnt_floor, returnPoints=False)
        defects = cv2.convexityDefects(cnt_floor, convex_floor)

        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt_floor[s][0])
                end = tuple(cnt_floor[e][0])
                far = tuple(cnt_floor[f][0])

                dst = self.distance(start, end)

                mannualy_contour.append(start)
                if dst >= 40:
                    mannualy_contour.append(far)
                mannualy_contour.append(end)

        mannualy_contour = np.array(mannualy_contour).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(debug, [mannualy_contour], 0, (0,255,0), 3)

        mask_floor = np.zeros(mask_floor.shape, dtype="uint8")
        mask_floor = cv2.fillPoly(mask_floor, pts = [mannualy_contour], color=(255,255,255))

        mask = cv2.bitwise_and(mask_overlay, mask_overlay, mask=mask_floor)
        mask_blue = cv2.bitwise_and(mask_blue, mask_blue, mask=mask_floor)

        masks = [mask, mask_blue]

        # Проходимся по маскам для нахождения пожаров и пострадавших
        for idx, m in enumerate(masks):
            contours = cv2.findContours(m, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)[-2]

            frame_vol = np.prod(frame.shape[0:2])

            # Фильтруем объекты по площади
            assert frame_vol != 0
            contours = list(filter(
                    lambda c: (cv2.contourArea(c) / frame_vol) >= self.fire_fraction and (cv2.contourArea(c) / frame_vol) < 0.2, 
                    contours))

            # Находим центры объектов в кадре
            pnt_img = []
            for cnt in contours:
                M = cv2.moments(cnt)

                if M["m00"] == 0:
                    continue

                pnt_img.append(
                        [int(M["m10"] / (M["m00"])),
                        int(M["m01"] / (M["m00"]))])

                cv2.circle(debug, tuple(pnt_img[-1]), 6, (255, 0, 0), 2)

                color = ((0,255,0) if idx == 0 else (0, 0, 255))
                cv2.drawContours(debug, [cnt], 0, color, 3) 

            # Находим координаты объекта, относительно aruco_map
            if len(pnt_img) > 0:
                pnt_img = np.array(pnt_img).astype(np.float64)
                pnt_img_undist = cv2.undistortPoints(pnt_img.reshape(-1, 1, 2), self.cm, self.dc, None, None).reshape(-1, 2).T
                ray_v = np.ones((3, pnt_img_undist.shape[1]))
                ray_v[:2, :] = pnt_img_undist
                ray_v /= np.linalg.norm(ray_v, axis=0)

                if self.tf_buffer is not None:
                    try:
                        transform = self.tf_buffer.lookup_transform("aruco_map", "main_camera_optical", rospy.Time())
                    except tf2_ros.ConnectivityException:
                        print("LookupException")
                        return None
                    except tf2_ros.LookupException:
                        print("LookupException")
                        return None

                    t_wb = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])

                    ray_v = np.array([unpack_vec(tf2_geometry_msgs.do_transform_vector3(Vector3Stamped(vector=Vector3(v[0], v[1], v[2])), transform)) for v in ray_v.T])
                    ray_o = t_wb

                    pnts = [intersect_ray_plane(v, ray_o) for v in ray_v]
                    [self.insert_fire(p[:2], idx) for p in pnts if p is not None]

        # Публикуем маркеры rviz и изображения дл отладки
        self.publish_markers()
        self.publish_markers_blue()
        self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug, "bgr8"))
        self.mask_overlay_pub.publish(self.cv_bridge.cv2_to_imgmsg(mask_floor, "mono8"))
