"""
patient_watcher.py
authors: jet, jack
"""

import rclpy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from my_robot_interfaces.msg import Person
from my_robot_interfaces.msg import PersonLandmark
from numpy import asarray
import mediapipe as mp
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, TimeSynchronizer

import cv2
import numpy as np  ### Import necessary libraries for Tiny YOLOv7

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        
        self.bridge = CvBridge()
        self.publisher1 = self.create_publisher(Person, 'person', 1)
        color_sub = Subscriber(self, Image, '/camera/color/image_rect_raw')
        depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')

        ts = TimeSynchronizer([color_sub, depth_sub], 10)
        ts.registerCallback(self.callback)

        ### Load Tiny YOLOv7 model
        self.weights = '/home/pmr/ros2_ws/src/patient_monitoring/patient_monitoring/yolo/yolov7-tiny.weights'
        self.config = '/home/pmr/ros2_ws/src/patient_monitoring/patient_monitoring/yolo/yolov7-tiny.cfg'
        self.model = cv2.dnn_DetectionModel(self.config, self.weights)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    def callback(self, msg: Image, msg_depth: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(msg_depth, msg_depth.encoding)

        img_h, img_w, c = cv_image.shape
        
        person_detected = Person()
        person_detected.x = -1
        person_detected.y = -1
        person_detected.w = -1
        person_detected.h = -1
        person_detected.depth = -1
        person_detected.x_min = -1
        person_detected.x_max = -1
        person_detected.y_min = -1
        person_detected.y_max = -1
        person_detected.image_width = img_w
        person_detected.image_height = img_h

        ### Replace MediaPipe code with Tiny YOLOv7
        classes, confidences, boxes = self.model.detect(cv_image, confThreshold=0.2, nmsThreshold=0.4)

        # Assuming the class_id of a person is 0, if not, change it accordingly
        person_class_id = 0
        print(classes, confidences)
        if len(classes) > 0 and len(confidences) > 0:
            for class_id, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                if class_id == person_class_id:
                    x, y, w, h = box
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    person_detected.x = int(x + w // 2)
                    person_detected.y = int(y + h // 2)
                    person_detected.w = int(w)
                    person_detected.h = int(h)
                    person_detected.x_min = int(x)
                    person_detected.x_max = int(x + w)
                    person_detected.y_min = int(y)
                    person_detected.y_max = int(y + h)

                    # Depth calculation
                    if(person_detected.x < img_w and person_detected.y < img_h):
                        person_detected.depth = 0
                        count = 0
                        for i in range(-3, 4, 1):
                            for j in range(-3, 4, 1):
                                if int(depth_image[person_detected.y + i, person_detected.x + j]) > 0:
                                    person_detected.depth = person_detected.depth + int(depth_image[person_detected.y + i, person_detected.x + j])
                                    count = count + 1
                        if count > 0:
                            person_detected.depth = person_detected.depth // count
                            line = '\rPerson detected! (%3d, %3d): %7.1f(mm).' % (person_detected.x, person_detected.y, depth_image[person_detected.y, person_detected.x])
                            print(line)

                    # Pose calculation
                    roi = cv_image[y:y+h, x:x+w]

                    image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    results = self.pose_estimator.process(image_rgb)
                    landmarks = results.pose_landmarks

                    if (landmarks):
                        landmarks = results.pose_landmarks.landmark
                        x_max = -1
                        y_max = -1
                        x_min = w
                        y_min = h
                        for i in range(len(landmarks) - 1):
                            landmark_msg = PersonLandmark()
                            landmark_msg.x = landmarks[i].x
                            landmark_msg.y = landmarks[i].y
                            landmark_msg.z = landmarks[i].z
                            landmark_msg.visibility = landmarks[i].visibility
                            person_detected.landmarks.append(landmark_msg)
                            x, y = int(landmarks[i].x * w), int(landmarks[i].y *h)
                            if 0 <= x and x <= w:
                                if x > x_max:
                                    x_max = x
                                if x < x_min:
                                    x_min = x
                            if 0 <= y and y <= h:
                                if y > y_max:
                                    y_max = y
                                if y < y_min:
                                    y_min = y
                    mp_drawing.draw_landmarks(
                       cv_image,
                       results.pose_landmarks,
                       mp_pose.POSE_CONNECTIONS,
        			   landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())            

        self.publisher1.publish(person_detected)
          
        cv2.imshow('cv_image', cv_image)
        cv2.waitKey(1)
def main(args=None):
    rclpy.init(args=args)

    node = MyNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()