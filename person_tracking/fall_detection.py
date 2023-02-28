import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os



def FallDetection(path):
    # For webcam input:
    cap = cv2.VideoCapture(path)
    _, frame = cap.read()
    h, w, c = frame.shape

    # intialise variables
    start_time = time.time()
    interval = 1

    prev_centroid = None
    prev_time = None
    prev_w = None
    prev_h = None

    folder_path = './data'
    file_name = 'pose_csv_file.csv'
    file_path = os.path.join(folder_path, file_name)


    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            success, image = cap.read()
            current_time = time.time()
            
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            landmarks = results.pose_landmarks
            
            
            if (landmarks):
                landmarks = results.pose_landmarks.landmark
                
                # drawing bounding box
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
                
                # Extract the coordinates of the shoulder, hip, and mid-point landmarks
                shoulder_left = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                shoulder_right = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                hip_left = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z)
                hip_right = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z)
                mid_point = ((shoulder_left[0] + shoulder_right[0])/2, (shoulder_left[1] + shoulder_right[1])/2)
                
                # Calculate the centroid of the person
                centroid = ((shoulder_left[0] + shoulder_right[0] + mid_point[0] + hip_left[0] + hip_right[0])/5, (shoulder_left[1] + shoulder_right[1] + mid_point[1] + hip_left[1] + hip_right[1])/40*9,(hip_left[2]+hip_right[2])/2)
                # Draw a circle at the centroid of the person
                cv2.circle(image, (int(centroid[0]*frame.shape[1]), int(centroid[1]*frame.shape[0])), 5, (0, 0, 255), -1)

                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                
                # fall detection
                base = (int(centroid[0]*frame.shape[1]) ,int(y_max))
                cv2.circle(image, base, 10, (0, 0, 255), -1)
                if (current_time - start_time) >= interval:
                    if prev_centroid is not None:
                        prev_distance = abs(int(prev_centroid[1]*frame.shape[0]) - y_max)
                        distance = abs(int(centroid[1]*frame.shape[0]) - y_max)
                        # print("prev",round(prev_distance,2))
                        # print("curr",round(distance,2))
                        if prev_distance != 0:
                            change = distance/prev_distance
                            if change<0.45:
                                print("fall")
                elif None not in (prev_w, prev_h):
                    if math.sqrt(pow(abs(centroid[0]-prev_centroid[0]), 2)+pow((centroid[1]-prev_centroid[1]), 2)) >= 0.5*math.sqrt(pow(bbox_w, 2)+pow(bbox_h, 2)):
                        pass
                    prev_centroid = centroid
                    start_time = current_time
                    prev_w = bbox_w
                    prev_h = bbox_h
            
            
            
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    data_folder = "data/videos/"
    videos = [file for file in os.listdir(data_folder) if file.endswith(".avi")]
    for vid in videos:
        FallDetection(vid)

if __name__ == "__main__":
    main()