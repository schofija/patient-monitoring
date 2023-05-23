import cv2
from ultralytics import YOLO
import time
import torch

def normalize(tensor, frame_shape):
    x = tensor[:, 0] / frame_shape[0]
    y = tensor[:, 1] / frame_shape[1]
    return torch.stack((x, y, tensor[:, 2]), dim=1)


def fall_detection(landmarks, frame_shape, start_time):
    for i in range(landmarks.shape[0]):
        person = landmarks[i]
        #print(person1, person1.shape)
        person = normalize(person, frame_shape)
        shoulder_right = person[2]
        shoulder_left = person[5]
        hip_right = person[8]
        hip_left = person[11]
        centroid = ((shoulder_left[0] + shoulder_right[0] + hip_left[0] + hip_right[0])/4, (shoulder_left[1] + shoulder_right[1] + hip_left[1] + hip_right[1])/4)
        base = (centroid[0], torch.min(person[:, 1]))
    #print(shoulder_right, shoulder_right.shape)

model = YOLO('yolov8n-pose.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#video_path = "football1.mp4"
cap = cv2.VideoCapture(0)
#writer = imageio.get_writer("output1.mp4", mode="I")
prev_centroid = None
prev_time = None
interval = 1
fall_count = 0
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        for result in results:
            keypoints = result.keypoints
            for i in range(keypoints.shape[0]):
                person = keypoints[i]
                #print(person1, person1.shape)
                person = normalize(person, frame.shape)
                shoulder_right = person[2]
                shoulder_left = person[5]
                hip_right = person[8]
                hip_left = person[11]
                centroid = ((shoulder_left[0] + shoulder_right[0] + hip_left[0] + hip_right[0])/4, (shoulder_left[1] + shoulder_right[1] + hip_left[1] + hip_right[1])/4)
                base = (centroid[0], torch.min(person[:, 1]))
                y_max = torch.max(person[:, 1])
                current_time = time.time()
                if (current_time - start_time) >= interval:
                    if prev_centroid is not None:
                        prev_distance = abs(int(prev_centroid[1] * frame.shape[0]))
                        distance = abs(int(centroid[1]*frame.shape[0]) - y_max)
                        if prev_distance != 0:
                            change = distance/prev_distance
                            if change < 0.45:
                                print("FALL")
                                fall_count +=1
                    prev_centroid = centroid
                    start_time = current_time
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print("FPS :", fps)
        print(fall_count)
        
        cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
        
        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        #writer.append_data(annotated_frame)         
        cv2.imshow("YOLOv8 Pose Estimation Demo", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(fall_count)