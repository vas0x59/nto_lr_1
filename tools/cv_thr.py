#!/usr/bin/env python3

import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('first')
# cv2.namedWindow('second')

cv2.createTrackbar('H Lower', 'first', 0, 179, nothing)
cv2.createTrackbar('H Higher', 'first', 179, 179, nothing)
cv2.createTrackbar('S Lower', 'first', 0, 255, nothing)
cv2.createTrackbar('S Higher', 'first', 255, 255, nothing)
cv2.createTrackbar('V Lower', 'first', 0, 255, nothing)
cv2.createTrackbar('V Higher', 'first', 255, 255, nothing)

# cv2.createTrackbar('H Lower', 'second', 0, 179, nothing)
# cv2.createTrackbar('H Higher', 'second', 179, 179, nothing)
# cv2.createTrackbar('S Lower', 'second', 0, 255, nothing)
# cv2.createTrackbar('S Higher', 'second', 255, 255, nothing)
# cv2.createTrackbar('V Lower', 'second', 0, 255, nothing)
# cv2.createTrackbar('V Higher', 'second', 255, 255, nothing)

# frame = cv2.imread('Screenshot from 2023-03-10 15-24-27.png')


cap = cv2.VideoCapture("/Users/vasily/Downloads/output_2.avi")

print(cap.isOpened())
_, frame = cap.read()

play = False

while(1):

    if play:
        _, frame = cap.read()

    cv2.imshow("frame", frame)
    
    # img = cv2.flip(frame, 1)

    # img = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hL = cv2.getTrackbarPos('H Lower', 'first')
    hH = cv2.getTrackbarPos('H Higher', 'first')
    sL = cv2.getTrackbarPos('S Lower', 'first')
    sH = cv2.getTrackbarPos('S Higher', 'first')
    vL = cv2.getTrackbarPos('V Lower', 'first')
    vH = cv2.getTrackbarPos('V Higher', 'first')

    LowerRegion = np.array([hL, sL, vL], np.uint8)
    upperRegion = np.array([hH, sH, vH], np.uint8)

    first_mask = cv2.inRange(hsv, LowerRegion, upperRegion)
    # first_mask = cv2.erode(first_mask, None, iterations=2)
    # first_mask = cv2.dilate(first_mask, None, iterations=2)

    # hL = cv2.getTrackbarPos('H Lower', 'second')
    # hH = cv2.getTrackbarPos('H Higher', 'second')
    # sL = cv2.getTrackbarPos('S Lower', 'second')
    # sH = cv2.getTrackbarPos('S Higher', 'second')
    # vL = cv2.getTrackbarPos('V Lower', 'second')
    # vH = cv2.getTrackbarPos('V Higher', 'second')

    # LowerRegion = np.array([hL, sL, vL], np.uint8)
    # upperRegion = np.array([hH, sH, vH], np.uint8)

    # second_mask = cv2.inRange(hsv, LowerRegion, upperRegion)
    # second_mask = cv2.erode(second_mask, None, iterations=2)
    # second_mask = cv2.dilate(second_mask, None, iterations=2)

    cv2.imshow("Masking first", first_mask)
    # cv2.imshow("Masking second", second_mask)
    # cv2.imshow("Masking result", cv2.bitwise_and(first_mask, second_mask))

    g = cv2.waitKey(1) & 0xFF
    if g == ord('q'):
        break
    elif g == ord(' '):
        play = not play


    

cv2.destroyAllWindows()