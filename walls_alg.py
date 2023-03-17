from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, NamedTuple, Callable
from enum import Enum
from copy import copy
import sklearn
from sklearn.cluster import DBSCAN
import itertools

from utils import *

def linesp_agg1(lines: List[LineSP]) -> LineSP:
    """
    Функция построения "среднего" отрезка 
    """
    ls = [make_dir_the_same(lines[0], l) for l in lines]
    mean_v = np.mean([l.v/np.linalg.norm(l.v) for l in ls], axis=0)
    mean_v /= np.linalg.norm(mean_v)
    mean_o = np.mean([l.p_tau(0.5) for l in ls], axis=0)
    l_tmp = LineSP(mean_o, mean_v, 1)
    a = min(ls, key=lambda l: param_of_nearest_pnt_on_line(l.p_tau(0), l_tmp)).p_tau(0)
    b = max(ls, key=lambda l: param_of_nearest_pnt_on_line(l.p_tau(1), l_tmp)).p_tau(1)

    a_t = param_of_nearest_pnt_on_line(a, l_tmp)
    b_t = param_of_nearest_pnt_on_line(b, l_tmp)

    v = l_tmp.p_t(b_t) - l_tmp.p_t(a_t)
    
    return LineSP(l_tmp.p_t(a_t), v/np.linalg.norm(v), np.linalg.norm(v))

# def linesp_agg2(lines: List[LineSP]) -> LineSP: # fit line points 
#     pass


# def frechet_distance(l1: LineSP, l2: LineSP) -> float:
#     pass

def distance_2(l1: LineSP, l2: LineSP)->float:
    return min(np.sum(np.power([*(l1.p0 - l2.p0), *(l1.p1 - l2.p1)], 2)), np.sum(np.power([*(l1.p0 - l2.p1), *(l1.p1 - l2.p0)], 2)))

def simple_distance(l1: LineSP, l2: LineSP) -> float:
    return np.arccos(l1.v.dot(l2.v) / np.linalg.norm(l1.v) / np.linalg.norm(l2.v)) + 3*np.linalg.norm(l1.p_tau(0.5) - l2.p_tau(0.5)) / (l1.len() + l2.len()) / 0.5


@dataclass
class Params:
    ldf: Callable[[LineSP, LineSP], float] # lines distances function (similarity)
    assign_to_prev_th: float
    connect_dist_th: float
    dbscan_eps: float 
    dbscan_min_samples: int = 1
    nms_th: float = 0.5


import math


def rotated_rect_from_line(l: LineSP, t: float) ->Tuple:
    """
    RotatedRect из LineSP
    """
    return (
        tuple(l.p_tau(0.5)), (l.len(), t), (180/np.pi)*math.atan2(l.v[1], l.v[0])
    )

def rect_overlap_metric(l1: LineSP, l2: LineSP, t: float = 0.3) -> float:
    """
    Метрика схожести линий
    """
    a1 = l1.len()*t
    a2 = l2.len()*t
    r1 = rotated_rect_from_line(l1, t)
    r2 = rotated_rect_from_line(l2, t)
    intr = cv2.rotatedRectangleIntersection(r1, r2)
    # print(intr)
    if intr[0] == 0:
        return 1
    elif len(intr[1].reshape(-1, 2))  >= 3:
        return 1 - cv2.contourArea(intr[1].reshape(-1, 2))/min(a1, a2)
    else:
        return 1

def assign_to_prev(hls: List[LineSP], pls: List[LineSP], params: Params) -> List[Optional[int]]: 
    out = [None for _ in hls]
    for i, hl in enumerate(hls):
        ds = list(map(lambda x: params.ldf(hl, x), pls))
        if len(ds) == 0:
            continue
        j = min(enumerate(ds), key=lambda x: x[1])[0]
        if ds[j] > params.assign_to_prev_th:
            continue
        out[i] = j
    return out


def create_new_pls_dbscan(hls: List[LineSP], params: Params) -> List[LineSP]:
    """
    Кластеризация линий (DBSCAN)
    """
    # cluster
    # linesp from cluster
    # done
    if len(hls) == 0:
        return []
    def metric(x, y):
        return params.ldf(LineSP(x[:2], x[2:4], x[4]), LineSP(y[:2], y[2:4], y[4]))

    X = np.array([
        [*l.p0, *l.v, l.t_max]
        for l in hls
    ])
    
    clustering = DBSCAN(eps=params.dbscan_eps, min_samples=params.dbscan_min_samples, metric=metric).fit(X)
    # clustering.
    n = clustering.labels_.max()
    clusters = [[] for i in range(n+1)]
    for hl, label in zip(hls, clustering.labels_):
        if label != -1:
            clusters[label].append(hl)
    # print("n:", n)
    
    return [linesp_agg1(c) for c in clusters]




def create_new_pls_nms(hls: List[LineSP], params: Params) -> List[LineSP]:
    bboxes = [[min(l.p0[0], l.p1[0]), min(l.p0[1], l.p1[1]), max(l.p0[0], l.p1[0]) - min(l.p0[0], l.p1[0]), max(l.p0[1], l.p1[1]) - min(l.p0[1], l.p1[1])] for l in hls]
    scores = [1 for _ in hls]
    ids = cv2.dnn.NMSBoxes(bboxes, scores, 0, params.nms_th).flatten()

    return [hls[i] for i in ids]

# def create_new_pls_simple(hls: List[LineSP], params: Params) -> List[LineSP]:
#     clusters = []
#     clusters_avg = []
#     # dists = np.zeros((len(hls), len(hls))) - 1
#     # for (i1, l1), (i2, l2) in itertools.permutations(enumerate(hls), 2):
#     #     dists[i1, i2] = params.ldf(l1, l2)
#     #     dists[i2, i1] = dists[i1, i2]
    
#     for h in hls:
#         if len(clusters) == 0:
#             clusters.append([h])
#             clusters_avg.append([h])
#             continue
#         # find min d
#         ds = list(map(lambda x: params.ldf(x, h), clusters_avg))
#         i, l = min(enumerate(ds), key=lambda x: x[1])
#         if ds[i] < params.dbscan_eps:
#             clusters[i].append(h)
#             clusters_avg[i] = linesp_agg1(clusters[i])
#         else:
            
    
#     return clusters_avg
            

def update_pls(pls_prev: List[LineSP], hls: List[LineSP], pls_hls_asg_h_index: List[List[int]], params: Params) -> List[LineSP]:
    # update pls using new hls
    pls_new = pls_prev.copy()
    for pls_i, hls_indexes in enumerate(pls_hls_asg_h_index):
        if len(hls_indexes) == 0:
            continue
        hls_i = [hls[j] for j in hls_indexes]
        pls_new[pls_i] = linesp_agg1(hls_i + [pls_new[pls_i]])

    return pls_new


class WallStates(Enum):
    FIXED = 0
    UPDATABLE = 1
    NOTUSED = 2
    
        
@dataclass
class Wall:
    line: LineSP
    state: WallStates

@dataclass
class StateObj:
    walls: List[Wall]
    current: int





# def inject_new(pls_prev_obj: Pls, pls_new: List[LineSP], params: Params) -> Pls:
#     dists_min = np.zeros((len(pls_new), len(pls_new))) - 1
#     dists_pn = np.zeros((len(pls_new), len(pls_new))) - 1 
#     for (i1, l1), (i2, l2) in itertools.permutations(enumerate(pls_new), 2):
#         s = [norm(l1.p0 - l2.p0), norm(l1.p1 - l2.p0), norm(l1.p1 - l2.p1), norm(l1.p0 - l2.p1)]
#         dists_min[i1, i2], dists_pn[i1, i2] = min(enumerate(s), key=lambda x: s[1])
#         dists_min[i2, i1] = dists_min[i1, i2]
#         dists_pn[i2, i1] = [0, 3, 2, 1][dists_pn[i1, i2]]
#     print("dists_min", dists_min)
#     print("dists_pn", dists_pn)
    
#     used = [False for _ in pls_new]
#     g: Dict[int, int] 


# def make_axis_parallel()

class WallsProcessor:
    def __init__(self, params: Params) -> None:
        self.params = params
        self.state_obj = StateObj([], -1)

    def proc_new(self, hls_clustered: List[LineSP], start_point: np.ndarray):
        # hls_clustered = create_new_pls_dbscan(hls, params)
        
        pls = [w.line for w in self.state_obj.walls]
        hls_pl_index = assign_to_prev(hls_clustered, pls, self.params)
        not_assigned = [i for i, j in enumerate(hls_pl_index) if j is None]
        print("not_assignet", not_assigned)
        pls_hls_asg_h_index = [[] for _ in pls]
        for i, j in enumerate(hls_pl_index): 
            if j is not None: pls_hls_asg_h_index[j].append(i)

        # assigned = [i for i, j in enumerate(hls_pl_index) if j is not None]

        # update
        for wall_id, hls_indexes in enumerate(pls_hls_asg_h_index):
            if len(hls_indexes) == 0 or self.state_obj.walls[wall_id].state == WallStates.FIXED:
                continue
            hls_i = [hls_clustered[j] for j in hls_indexes]
            self.state_obj.walls[wall_id].line = make_dir_the_same(self.state_obj.walls[wall_id].line, linesp_agg1(hls_i + [self.state_obj.walls[wall_id].line]))
            



        # add new 
        pls_candidates = [hls_clustered[i] for i in not_assigned]
        
        # init
        if len(self.state_obj.walls) == 0:
            crt: Wall = Wall(LineSP(start_point, np.array([-1, -1]), 1).reversed(), WallStates.UPDATABLE)
        else:
            crt: Wall = self.state_obj.walls[self.state_obj.current]
        
        ds = [-1 for _ in pls_candidates]
        to_rev = [False for _ in pls_candidates]
        
        for i, pln in enumerate(pls_candidates):
            j, ds[i] = min( enumerate([norm(pln.p0 -crt.line.p1), norm(pln.p1 -crt.line.p1)]), key=lambda x: x[1])
            to_rev[i] = j == 1

        ids = range(0, len(pls_candidates))
        if len(self.state_obj.walls) != 0:
            ids = list(filter(lambda i: ds[i] < self.params.connect_dist_th and math.acos(abs(np.dot(pls_candidates[i].v, crt.line.v)/norm(crt.line.v)/norm(pls_candidates[i].v))) > math.radians(60), ids))
        print("to_rev", to_rev)
        if len(ids) > 0:
            id_to_connect = min(ids, key=lambda i: ds[i])
            l_to_connect = pls_candidates[id_to_connect].reversed() if to_rev[id_to_connect] else pls_candidates[id_to_connect]
            crt.state = WallStates.FIXED
            self.state_obj.walls.append(Wall(l_to_connect, WallStates.UPDATABLE))
            self.state_obj.current = len(self.state_obj.walls) - 1

        # return inject_new(pls_updated, pls_new, params)
        # return Pls(pls=pls_updated + pls_candidates, g=dict())
        






