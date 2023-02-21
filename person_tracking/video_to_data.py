import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import csv
import time
import math

def find_centroid(lm):
  points = [lm[11], lm[12], lm[23], lm[24]]
  l = len(points)
  x = [p.x for p in points]
  y = [p.y for p in points]
  z = [p.z for p in points]
  return (sum(x)/l), (sum(y)/l), (sum(z)/l)

# only works with bbox
# def detect_fall(curr_pose, prev_pose):
#   label = 0
#   if w >= 1.2*h:
#     label = 1
#   elif np.sqrt(np.power(abs(curr_x - old_x), 2)+np.power(curr_y - old_y))

def get_important_landmarks(lm):
  # nose = [lm[0]]
  # arms = lm[11:17]
  # base = lm[23:29]
  features = []
  kps = [0,11,12,13,14,15,16,23,24,25,26,27,28]
  for k in kps:
    features.append(lm[k].x)
    features.append(lm[k].y)
    features.append(lm[k].z)
    features.append(lm[k].visibility)
  return features

def get_video_files(data_folder):
  files = [file for file in os.listdir(data_folder) if file.endswith(".avi")]
  # format: '{data_foler_path}/'
  # files = []
  # for file in os.listdir(data_folder):
  #   if file.endswith(".avi"):
  #     files.append(file)
  return files



def get_video_data(mp_pose, video, folder):
  video_data = []
  #loop through Video inputs:
  path = folder + video
  
  cap = cv2.VideoCapture(path)
  _, frame = cap.read()
  #frame = cv2.resize(frame, (256,256))
  
  h, w, c = frame.shape
  frame_count = 0
  
  with mp_pose.Pose(
    static_image_mode = True,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    ) as pose:
    while cap.isOpened():
      success, image = cap.read()
      frame_count +=1

      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      # using a greyscale picture, also for faster detection
      #image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      results = pose.process(image)
      landmarks = results.pose_landmarks  
      if (landmarks):
        landmarks = results.pose_landmarks.landmark
        features = get_important_landmarks(landmarks)
        # if frame_count == 0:
        #   label = 0
        # label = detect_fall(landmarks, prev_landmarks)
        # features.append(label)
        video_data.append(features)
      # prev_landmarks = landmarks
      if cv2.waitKey(5) & 0xFF == 27:
        break

  cap.release()
  cv2.destroyAllWindows()
  print(len(video_data))
  return video_data

def main():
  mp_pose = mp.solutions.pose

  #Get videos
  data_folder = "data/videos/"
  src_videos = get_video_files(data_folder)

  count = 0
  for src in src_videos:
    time_start = time.time()
    print(f"Video {count} begun analysis")
    csv_name = "pose_data_video_" + str(count) + ".csv" 
    data = get_video_data(mp_pose, src, data_folder)
    print("Analysis time:", (time.time() - time_start))
    count+=1
    with open(csv_name, 'w', newline='') as f:
      write = csv.writer(f)
      write.writerows(data)
  
if __name__ == "__main__":
  main()