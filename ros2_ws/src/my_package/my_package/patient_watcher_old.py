import rclpy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from my_robot_interfaces.msg import Person
from numpy import asarray
import mediapipe as mp
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class ImageSubscriber(Node):
    def __init__(self):
        self.depth = None
        self.node = rclpy.create_node('image_subscriber')
        self.subscription_2 = self.node.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            1)
        self.subscription_1 = self.node.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.image_callback,
           1)
        self.bridge = CvBridge()
        self.publisher1 = self.node.create_publisher(Person, 'person', 1)
         #In case anyone needs a version of the topic with buffers:
        self.publisher2 = self.node.create_publisher(Person, 'buffered_person', 10)

    def image_callback(self, msg, msg_depth):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = self.bridge.imgmsg_to_cv2(msg_depth, msg_depth.encoding)
        # Save current depth image so we aren't using a newer one when we go to get depth for this image
        #depth_image = asarray(self.depth)
        #image = np.array(cv_image)
        h, w, c = cv_image.shape
        
        person_detected = Person()
        person_detected.x = -1
        person_detected.y = -1
        person_detected.w = -1
        person_detected.h = -1
        person_detected.depth = 0

        person_detected.landmarks = []
        
        # Detect objects in the image using mediapipe
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

               	results = pose.process(cv_image)
               	landmarks=results.pose_landmarks

               	if (landmarks):
                    landmarks = results.pose_landmarks.landmark
                    x_max = -1
                    y_max = -1
                    x_min = w
                    y_min = h
                    for i in range(len(landmarks)):
                        #person_detected.landmarks[i].x = landmarks[i].x
                        #person_detected.landmarks[i].y = landmarks[i].y
                        #person_detected.landmarks[i].z = landmarks[i].z
                        #person_detected.landmarks[i].visibility = landmarks[i].visibility
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

                    # depth_x and depth_y are coordinates we want to get depth from
                    #depth_x, depth_y
                    # person_detected.depth is the actual depth
                    person_detected.depth = 0
#DEPTH GRABBING BLOCK HERE
                    #print(x_min, x_max, y_min, y_max)
                    person_detected.x = (x_max + x_min)//2
                    person_detected.y = (y_max + y_min)//2
                    person_detected.w = x_max - x_min
                    person_detected.h = y_max - y_min
                    person_detected.depth = 0

                    cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
               	    cv_image.flags.writeable = True
               #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
               	mp_drawing.draw_landmarks(
                       cv_image,
                       results.pose_landmarks,
                       mp_pose.POSE_CONNECTIONS,
        			   landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
						 
                self.publisher1.publish(person_detected)
                self.publisher2.publish(person_detected)
                
               	#cv2.imshow('cv_image', cv_image)
               	#cv2.waitKey(1)



        # Draw BBoxes

        #self.publisher.publish(person_detected)
        
        # Display the processed image
        #cv2.imshow('cv_image', cv_image)
        #cv2.waitKey(1)

    def depth_callback(self, msg):
        #print("here!")
        #self.depth = msg
        print(person.x)
        cv_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        #indices = np.array(np.where(cv_image == cv_image[cv_image > 0].min()))[:,0]
        ##pix = (indices[1], indices[0])
        ##self.pix = pix
        pix = (212, 120)
        line = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (pix[0], pix[1], cv_image[pix[1], pix[0]])
        print(line)
        cv2.rectangle(cv_image, (10, 30), (20, 40), (0, 255, 255), 2)
        cv2.imshow('cv_image', cv_image)
        cv2.waitKey(1)


def main():
    rclpy.init()
    image_sub = ImageSubscriber()
    rclpy.spin(image_sub.node)

if __name__ == '__main__':
    main()
