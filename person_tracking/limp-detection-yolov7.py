import cv2
import time
import torch
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt


def calculate_angle(p1, p2, p3):
    # Calculate the angle formed by three points
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def calculate_leg_angles(landmarks, frame_shape):
    # Calculate the angles between the legs
    x_vals = landmarks[::3]
    x_vals = [(x / frame_shape[1]) for x in x_vals] #normalized to frames shape
    y_vals = landmarks[1::3]
    y_vals = [(y / frame_shape[0]) for y in y_vals] #normalized to frames shape
    knee_left = (x_vals[12], y_vals[12])
    knee_right = (x_vals[9], y_vals[9])

    ankle_left = (x_vals[13], y_vals[13])
    ankle_right = (x_vals[10], y_vals[10])

    hip_right = (x_vals[8], y_vals[8])
    hip_left = (x_vals[11], y_vals[11])
    left_leg_angle = calculate_angle(hip_left, knee_left, ankle_left)
    right_leg_angle = calculate_angle(hip_right, knee_right, ankle_right)
    return left_leg_angle, right_leg_angle

def limp_dectionV1(landmarks):
    # Check the difference in angles between the left and right legs
    left_leg_angle, right_leg_angle = calculate_leg_angles(landmarks)
    angle_diff = abs(left_leg_angle - right_leg_angle)
    if angle_diff > 10: # Threshold for significant angle difference
        return True
    else:
        return False

def limp_dectionV2(landmarks, frame_shape):

    x_vals = landmarks[::3]
    x_vals = [(x / frame_shape[1]) for x in x_vals] #normalized to frames shape
    y_vals = landmarks[1::3]
    y_vals = [(y / frame_shape[0]) for y in y_vals] #normalized to frames shape
    ankle_left = (x_vals[13], y_vals[13])
    ankle_right = (x_vals[10], y_vals[10])

    distance = math.dist(ankle_right, ankle_left)
    if distance < 0.9:
        print("Limp detected")
        return True


@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="0",device='cpu',view_img=True,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #device = select_device(device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    old_pose = None
    
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else :
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height

        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

        while(cap.isOpened): #loop until cap opened or video not complete
        
            print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture
            
            if ret: #if success is true, means frame exist
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)
                #print(output_data)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)
                # print(f"Output to keypoints {output}")
                # print(f"Keypoints maybe{output[0,7:].T}")
                print(f"Len keypoints {len(output[0,7:].T)} ")
                landmarks = output[0,7:].T # list of len 51. 17 kpts x, y, vis
                #print(output_data)
                print(f"Landmarks {landmarks}")
                print(f"Landmarks shape {landmarks.shape}")
                #fall detection
                if len(landmarks) == 51: # If we have pose data
                    print("Limp detection")
                    is_limpingv1 = limp_dectionV1(landmarks, frame.shape)
                    is_limpingv2 = limp_dectionV2(landmarks, frame.shape)

                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            #print(f"Keypoints {kpts.shape}")
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                        orig_shape=im0.shape[:2])

                
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                
                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(im0)  #writing the video frame

            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        
        #plot the comparision graph
        plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run() #**vars(opt)

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
