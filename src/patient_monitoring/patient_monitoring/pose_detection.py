# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19

@authors: aadi, jack
"""

import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import mediapipe as mp
import math
import time
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class PoseDetectionNode:
    def __init__(self):
        self.start_time = time.time()
        self.node = rclpy.create_node('pose_detection')
        
        self.subscription_1 = self.node.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.image_callback,
            1)
        
        self.bridge = CvBridge()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w, c = cv_image.shape
        l = []
        
        interval = 1

        folder_path = '/home/pmr/patient_monitoring'
        file_name = 'pose_csv_file.csv'
        file_path = os.path.join(folder_path, file_name)
        
        # Detect objects in the image using mediapipe
        with mp_pose.Pose(
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8) as pose:

                current_time = time.time()
                if not os.path.exists(file_path):
                    # Create new file with header row
                    with open(file_path, 'w', newline='') as csv_file:
                        writer = csv.writer(csv_file)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                cv_image.flags.writeable = False
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                results = pose.process(cv_image)
                landmarks = results.pose_landmarks
                
                if (current_time - self.start_time) >= interval:
                    landmarks = results.pose_landmarks
                    if (landmarks):
                        landmarks = results.pose_landmarks.landmark
                        for i in range(0,len(landmarks)):
                            # if (current_time - start_time) >= interval:
                            temp =  math.sqrt(abs(landmarks[i].x**2 - landmarks[i].y**2))
                            l.append(temp)
                        print(l)
                        with open(file_path, 'a', newline='') as csv_file:
                             writer = csv.writer(csv_file)
                             writer.writerow(l)
                        start_time = current_time
                        l = []

                        x_max = 0
                        y_max = 0
                        x_min = w
                        y_min = h    
                        for lm in landmarks:
                           x, y = int(lm.x * w), int(lm.y * h)
                           if x > x_max:
                               x_max = x
                           if x < x_min:
                               x_min = x
                           if y > y_max:
                               y_max = y
                           if y < y_min:
                                   y_min = y

                        cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw the pose annotation on the image.
                cv_image.flags.writeable = True
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    cv_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())   
                
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(cv_image, 1))
               	cv2.waitKey(1)

def main():
    rclpy.init()
    pose_detect = PoseDetectionNode()
    rclpy.spin(pose_detect.node)

if __name__ == '__main__':
    main()
