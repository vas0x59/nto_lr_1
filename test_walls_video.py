import cv2
import numpy as np
import yaml

import walls 

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

cm, dc = (lambda x: (x["K"], x["D"][0]))(yaml.load(open("test.yaml", "r").read(), Loader=yaml.FullLoader))
cm /= 2

wd = walls.WallDetector(cm, dc, image_wh=(320, 240))


cap = cv2.VideoCapture("/Users/vasily/Downloads/output_2.avi")

i = 0

while cap.isOpened():
    r, frame = cap.read()
    if not r:
        break
    
    cv2.imshow("frame", frame)

    wd.on_frame(frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord(' '):
        cv2.imwrite(f"image{i}.png", frame)
        i+=1
cap.release()


