import cv2
from ultralytics import YOLO
import numpy as np

def is_limping_v1(landmarks):
    # Check distance between ankles
    left_ankle = landmarks[11]
    right_ankle = landmarks[14]
    distance = ((left_ankle[0] - right_ankle[0])**2 + (left_ankle[1] - right_ankle[1])**2)**0.5
    if distance < 0.9:
        return True
    else:
        return False
def calculate_leg_angles(landmarks):
    # Calculate the angles between the legs
    left_hip = landmarks[11]
    left_knee = landmarks[13]
    left_ankle = landmarks[15]
    right_hip = landmarks[12]
    right_knee = landmarks[14]
    right_ankle = landmarks[16]
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    return left_leg_angle, right_leg_angle

def calculate_angle(p1, p2, p3):
    # Calculate the angle formed by three points
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def is_limping_v2(landmarks):
    # Check the difference in angles between the left and right legs
    left_leg_angle, right_leg_angle = calculate_leg_angles(landmarks)
    angle_diff = abs(left_leg_angle - right_leg_angle)
    if angle_diff > 10: # Threshold for significant angle difference
        return True
    else:
        return False
    
def main():

    # Load the YOLOv8 model
    model = YOLO('yolov8n-pose.pt')

    limp_v1_count = 0
    limp_v2_count = 0

    # Open the video file
    #video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Get Boxes
            boxes = results[0].boxes
            #print(boxes)
            box = boxes[0]
            #print(box)
            if (is_limping_v1):
                limp_v1_count +=1
                print("Limp detected (V1)")
            
            if (is_limping_v2):
                limp_v2_count +=1
                print("Limp detected (V2)")
            

            # Visualize the results on the frame
            annotated_frame = results[0].plot(labels=True, boxes=True)


            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()