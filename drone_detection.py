from __future__ import print_function
from TelloSDKPy.djitellopy import Tello
import time
import sys
import cv2
import argparse
import numpy as np
import math

CENTER_X = 960/2
CENTER_Y = 720/2
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
TAKEOFF = 1

#Get angle between two points 
def get_angle(start, end):
    angle = int(math.atan2((start[1] - end[1]), (start[0] - end[0])) * 180 / math.pi)
    return angle

#Get distance between two points
def get_distance(start, end):
    distance = int(math.sqrt((end[0] - start[0])**2+(end[1] - start[1])**2))
    return distance

#Check a list
def check_if_all_none(list_of_elem):
    result = True
    for elem in list_of_elem:
        if elem is not None:
            return False
    return result

'''
Head - 0, Neck - 1, Right Shoulder - 2, Right Elbow - 3, Right Wrist - 4,
Left Shoulder - 5, Left Elbow - 6, Left Wrist - 7, Right Hip - 8,
Right Knee - 9, Right Ankle - 10, Left Hip - 11, Left Knee - 12,
Left Ankle - 13, Chest - 14, Background - 15
'''

#Follow user
def follow_head(points):
    up_down_velocity = 0
    right_left_velocity = 0
    front_back_velocity = 0
    raw_velocity = 0
    
    #Center of drone and neck
    if(points[1] != None and points[2] != None and points[5] != None and points[3] != None):
        distance_x = CENTER_X - int(points[1][0])
        distance_y = CENTER_Y - int(points[1][1])
        distance_shoulder = get_distance(points[2], points[5])

        print(distance_x)
        print(distance_y)
        
        if distance_shoulder < 250:
            front_back_velocity = 10
        elif distance_shoulder > 350:
            front_back_velocity = -10
        else:
            print('all is good')

        if (distance_x > 150):
            right_left_velocity = -10
        elif (distance_x < -150):
            right_left_velocity = 10
        else:
            print('all is good')

        if (distance_y > 150):
            up_down_velocity = 10
        elif (distance_y < -150):
            up_down_velocity = -10
        else:
            print('all is good')

        tello.send_rc_control(int(right_left_velocity), int(front_back_velocity), int(up_down_velocity), int(raw_velocity))

#Detect pose if arm left straight
def is_left_straight(points):
        left = False
        if points[5] and points[6] and points[7]:
            shoulder_angle = get_angle(points[5], points[6])            
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360            
            if 140 < shoulder_angle < 190:
                elbow_angle = get_angle(points[6], points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                if abs(elbow_angle - shoulder_angle) < 30:
                    left = True
        if left:
            return True
        else:
            return False

#Detect pose if arm left angle
def is_left_sq(points):
        left = False
        if points[5] and points[6] and points[7]:
            shoulder_angle = get_angle(points[5], points[6])            
            if shoulder_angle < 0:
                shoulder_angle = shoulder_angle + 360            
                elbow_angle = get_angle(points[6], points[7])
                if elbow_angle < 0:
                    elbow_angle = elbow_angle + 360
                if 70 < abs(elbow_angle - shoulder_angle) < 110:
                    left = True
        if left:
            return True
        else:
            return False

#Detect pose if arm right straight
def is_right_straight(points):
        right = False
        if points[2] and points[3] and points[4]:
            shoulder_angle = get_angle(points[2], points[3])
            if -10 < shoulder_angle < 40:
                elbow_angle = get_angle(points[3], points[4])
                if abs(elbow_angle - shoulder_angle) < 30:
                    right = True
        if right:
            return True
        else:
            return False

#Detect pose if arm right angle
def is_right_sq(points):
        right = False
        if points[2] and points[3] and points[4]:
            shoulder_angle = get_angle(points[2], points[3])
            elbow_angle = get_angle(points[3], points[4])
            if 70 < abs(elbow_angle - shoulder_angle) < 110:
                right = True
        if right:
            return True
        else:
            return False

#Detect pose if right arm cross
def is_right_cross(points):
    right = False
    if points[2] and points[3] and points[4] :
        shoulder_right_angle = get_angle(points[2], points[3])
        elbow_right_angle = get_angle(points[3], points[4])
        if shoulder_right_angle < 0:
            shoulder_right_angle = shoulder_right_angle + 360  
        if 255 < shoulder_right_angle < 285 and 125 < elbow_right_angle < 155:
            right = True
    if right:
        return True
    else:
        return False

#Detect pose if right arm up
def is_right_up(points):
    right = False
    if points[2] and points[3]:
        shoulder_right_angle = get_angle(points[2], points[3])
        if 50 < shoulder_right_angle < 80:
            right = True
    if right:
        return True
    else:
        return False

#Detect pose if left arm up
def is_left_up(points):
    right = False
    if points[5] and points[6]:
        shoulder_right_angle = get_angle(points[5], points[6])
        if 50 < shoulder_right_angle < 80:
            right = True
    if right:
        return True
    else:
        return False

#Detect pose if left arm cross
def is_left_cross(points):
    left = False
    if points[5] and points[6] and points[7]:
        shoulder_left_angle = get_angle(points[5], points[6])
        elbow_left_angle = get_angle(points[6], points[7])
        if shoulder_left_angle < 0:
            shoulder_left_angle = shoulder_left_angle + 360  
        if 255 < shoulder_left_angle < 285 and 25 < elbow_left_angle < 55:
            left = True
    if left:
        return True
    else:
        return False 


#Detect poses
def detect_pose(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inWidth = 168
    inHeight = 168
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    frame = cv2.circle(frame, (int(CENTER_X), int(CENTER_Y)), 12, (0, 0, 255), 8)
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > 0.05 :
            cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            points.append((int(x), int(y)))
        else :
            points.append(None)
    
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
    
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (255, 255,0), 2)
            cv2.circle(frame, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

    detect_pose = 0

    if is_left_cross(points):
        print('arm left cross')
        tello.send_rc_control(0, 0, 0, -20)
        detect_pose = 1
    if is_right_cross(points):
        print('arm right cross')
        tello.send_rc_control(0, 0, 0, -20)
        detect_pose = 1
    if is_left_straight(points):
        print('arm flat left detected')
        tello.send_rc_control(20, 0, 0, 0)
        detect_pose = 1
    if is_right_straight(points):
        print('arm flat right detected')
        tello.send_rc_control(-20, 0, 0, 0)
        detect_pose = 1
    if is_left_sq(points):
        print('arm left square')
        tello.send_rc_control(20, 0, 0, 0)
        detect_pose = 1
    if is_right_sq(points):
        print('arm right square')
        tello.send_rc_control(0, 0, 0, -20)
        detect_pose = 1
    if is_right_up(points):
        print("right up")
        tello.send_rc_control(0, 0, 20, 0)
    if is_left_up(points):
        print("left up")
        tello.send_rc_control(0, 0, -20, 0)
    if detect_pose == 0:
        follow_head(points)
   
    cv2.imshow("Frame", frame)


#Connect drone and stream on
tello = Tello()
tello.connect()
tello.takeoff()
tello.streamon()

#Model to use Caffe and MPII
protoFile ="data/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "data/pose_iter_160000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

while True:
    #Read tello frame
    tello_frame = tello.get_frame_read().frame
    print('battery')
    print(tello.battery)
    if tello_frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detect_pose(tello_frame)
    k = cv2.waitKey(33)
    if k==27: 
        break
    elif k==-1:
        continue
    else:
        print(k)

tello.land()
cv2.destroyAllWindows()