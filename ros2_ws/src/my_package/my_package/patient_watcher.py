import rclpy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from my_robot_interfaces.msg import Person
from numpy import asarray
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class ImageSubscriber:
    def __init__(self):
        self.depth = None
        self.node = rclpy.create_node('image_subscriber')
        self.subscription_2 = self.node.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',     #Get correct name for this
            self.depth_callback,
            1)
        self.subscription_1 = self.node.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.image_callback,
            1)
        self.bridge = CvBridge()
        self.publisher = self.node.create_publisher(Person, 'person', 1)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        #image = np.array(cv_image)
        h, w, c = cv_image.shape
        
        person_detected = Person()
        person_detected.x = -1
        person_detected.y = -1
        person_detected.w = -1
        person_detected.h = -1
        person_detected.depth = 0
        
        # Detect objects in the image using mediapipe
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

               	results = pose.process(cv_image)
               	landmarks=results.pose_landmarks

               	if (landmarks):
                    landmarks = results.pose_landmarks.landmark
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in landmarks:
                        x, y = int(lm.x * w), int(lm.y *h)
                        if x > x_max:
                           x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    print(x_min, x_max, y_min, y_max)
                    person_detected.x = (x_max + x_min)//2
                    #person_detected.y = -1.
                    #person_detected.w = -1.
                    #person_detected.h = -1.
                    #person_detected.depth = 0
                    cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

               	cv_image.flags.writeable = True
               #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
               	mp_drawing.draw_landmarks(
                       cv_image,
                       results.pose_landmarks,
                       mp_pose.POSE_CONNECTIONS,
        			   landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
						 
                self.publisher.publish(person_detected)
                
               	cv2.imshow('cv_image', cv_image)
               	cv2.waitKey(1)



        # Draw BBoxes

        #self.publisher.publish(person_detected)
        
        # Display the processed image
        #cv2.imshow('cv_image', cv_image)
        #cv2.waitKey(1)

    def depth_callback(self, msg):
        self.depth = msg

def main():
    rclpy.init()
    image_sub = ImageSubscriber()
    rclpy.spin(image_sub.node)

if __name__ == '__main__':
    main()
