# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:17:13 2023

@author: user
"""

import rclpy
from geometry_msgs.msg import Twist
from my_robot_interfaces.msg import Person
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Predefined frame width/height
FRAME_WIDTH = 424
FRAME_HEIGHT = 240

class FallDetector():
    def __init__(self):
        self.start_time = time.time()
        self.interval = 1.
        self.prev_centroid = None
        self.prev_time = None
        self.node = rclpy.create_node('fall_detection')
        self.subscription = self.node.create_subscription(
            Person,
            'person',
            self.falldetector_callback,
            1)
    def falldetector_callback(self, msg):
        current_time = time.time()

        landmarks = msg.landmarks
        y_max = msg.y_max

        if (landmarks):
            print("in if (landmarks): ...")
            shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            hip_left = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z)
            hip_right = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z)
            mid_point = ((shoulder_left[0] + shoulder_right[0])/2, (shoulder_left[1] + shoulder_right[1])/2)

            # Calculate the centroid of the person
            centroid = ((shoulder_left[0] + shoulder_right[0] + mid_point[0] + hip_left[0] + hip_right[0])/5, (shoulder_left[1] + shoulder_right[1] + mid_point[1] + hip_left[1] + hip_right[1])/40*9,(hip_left[2]+hip_right[2])/2)
            # Draw a circle at the centroid of the person
            #cv2.circle(image, (int(centroid[0]*frame.shape[1]), int(centroid[1]*frame.shape[0])), 5, (0, 0, 255), -1)

            # fall detection
            #base = (int(centroid[0]*frame.shape[1]) ,int(y_max))
            base = (int(centroid[0]*FRAME_WIDTH) ,int(y_max))
           # cv2.circle(image, base, 10, (0, 0, 255), -1)
            if (current_time - self.start_time) >= self.interval:
                print("1 second interval check") 
                if self.prev_centroid is not None:
                    print("if self.prev_centroid is not None...")
                    #prev_distance = abs(int(self.prev_centroid[1]*frame.shape[0]) - y_max)
                    prev_distance = abs(int(self.prev_centroid[1]*FRAME_HEIGHT) - y_max)
                    #distance = abs(int(centroid[1]*frame.shape[0]) - y_max)
                    distance = abs(int(centroid[1]*FRAME_HEIGHT) - y_max)
                    # print("prev",round(prev_distance,2))
                    # print("curr",round(distance,2))
                    if prev_distance != 0:
                        print("if prev_distance != 0")
                        change = distance/prev_distance
                        if change<0.45:
                            print("fall")
                        else:
                            print("no fall")
                
                self.prev_centroid = centroid
                self.start_time = current_time

def main():
    rclpy.init()
    falldetector_sub = FallDetector()
    rclpy.spin(falldetector_sub.node)

if __name__ == '__main__':
    main()
