# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 03:31:56 2023

@author: aadi
"""

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


# For webcam input:
cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape
l = []

# intialise variables
start_time = time.time()
interval = 1

folder_path = 'C:/Users/user'
file_name = 'pose_csv_file.csv'
file_path = os.path.join(folder_path, file_name)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    current_time = time.time()
    # Check if file exists in folder
    if not os.path.exists(file_path):
        # Create new file with header row
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    landmarks = results.pose_landmarks
    
    
    if (current_time - start_time) >= interval:
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
               
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # mp_drawing.draw_axis(image, rotation, translation)    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()