from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

from sensor_msgs.msg import CameraInfo, Image

from typing import Tuple, Optional

n_plane = np.array([0, 0, 1])
p_plane = np.array([0, 0, 0])

def unpack_vec(v):
    return np.array([v.vector.x, v.vector.y, v.vector.z])


def camera_cfg_cvt(msg: CameraInfo) -> Tuple[np.ndarray, np.ndarray]:
    return (np.reshape(np.array(msg.K, dtype="float64"), (3, 3)), np.array(msg.D, dtype="float64"))

def _R_from_quat(q):
    qw, qx, qy, qz = q
    return [
        [qw**2 + qx**2 - qy**2 - qz**2, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), qw**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qw*qx + qy*qz), qw**2 - qx**2 - qy**2 + qz**2]
    ]
def np_R_from_quat(q):
    return np.array(_R_from_quat(q))


def ray_from_camera(pnt_img: np.ndarray, camera_mtx: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    pnt_img_undist = cv2.undistortPoints(np.array([pnt_img]).reshape(2, 1), camera_mtx, dist_coeffs, None, camera_mtx).reshape(2)
    ph = np.ones((3, 1)).astype(np.float64)
    ph[:2, 0] = pnt_img.astype(np.float64)
    
    ray_vector_camera = np.linalg.pinv(camera_mtx) @ ph
    ray_vector_camera /= np.linalg.norm(ray_vector_camera)

    return ray_vector_camera.reshape(3)


def intersect_ray_plane(ray_v, ray_o) -> Optional[np.ndarray]:
    a = n_plane.dot(ray_v)
    if a == 0:
        return None
    
    d = (p_plane - ray_o).dot(n_plane) / a

    return ray_o + d * ray_v



norm = np.linalg.norm

@dataclass
class LineSP: # t in [0 t_max]
    """
    Класс параметрического представления отрезка 
    
    """
    p0: np.ndarray
    v: np.ndarray
    t_max: float
    @property
    def p1(self):
        return self.v*self.t_max + self.p0
    def p_t(self, t):
        return self.v*t + self.p0
    def p_tau(self, t):
        return self.p_t(t*self.t_max)
    def len(self):
        return np.linalg.norm(self.v)*self.t_max
    @classmethod
    def from_2pnt(cls, a: np.ndarray, b: np.ndarray) -> LineSP:
        return cls(a, b-a, 1)
    def reversed(self) -> LineSP:
        return LineSP(self.p_tau(1), -self.v.copy(), self.t_max)
    def left_norm(self) -> np.ndarray:
        return np.array([-self.v[1], self.v[0] ])
    def right_norm(self) -> np.ndarray:
        return np.array([self.v[1], -self.v[0]])

def param_of_nearest_pnt_on_line(a: np.ndarray, l: LineSP) -> float:
    """
    Расчет параметра точки на линии ближайшей к данной 
    """
    b = a - l.p0
    return np.dot(b, l.v) / np.dot(l.v, l.v)

def make_dir_the_same(l_a: LineSP, l_b: LineSP) -> LineSP:
    """
    make l_b (l_b ^^ l_a)
    """
    if np.dot(l_a.v, l_b.v) >= np.dot(l_a.v, -l_b.v):
        return l_b
    return l_b.reversed()


def distance_to_linesp(a: np.ndarray, l: LineSP) -> float:
    t = np.clip(param_of_nearest_pnt_on_line(a, l), 0, l.t_max)
    return norm(l.p_t(t) - a)

def distance_to_line(a: np.ndarray, l: LineSP) -> float:
    t = param_of_nearest_pnt_on_line(a, l)
    return norm(l.p_t(t) - a)
