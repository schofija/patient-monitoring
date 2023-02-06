import rclpy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSubscriber:
    def __init__(self):
        self.node = rclpy.create_node('image_subscriber')
        self.subscription = self.node.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

        # Load YOLO
        self.net = cv2.dnn.readNetFromDarknet('/home/pmr/ros2_ws/src/my_package/my_package/yolov3.cfg', '/home/pmr/ros2_ws/src/my_package/my_package/yolov3.weights')
        self.classes = []
        with open('/home/pmr/ros2_ws/src/my_package/my_package/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def image_callback(self, msg):
        # Convert the image data from ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        #image = np.array(cv_image)

        # Image detection with Yolo
        # Detect objects in the image
        blob = cv2.dnn.blobFromImage(cv_image, 1 / 255.0, (416, 416), [0, 0, 0], swapRB=True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    x, y, w, h = (detection[0:4] * np.array([cv_image.shape[1], cv_image.shape[0], cv_image.shape[1], cv_image.shape[0]])).astype('int')
                    cv2.rectangle(cv_image, (x-w//2, y-h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                    label = '{}: {:.2f}'.format(self.classes[class_id], confidence)
                    cv2.putText(cv_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display the processed image
        cv2.imshow('cv_image', cv_image)
        cv2.waitKey(1)

def main():
    rclpy.init()
    image_sub = ImageSubscriber()
    rclpy.spin(image_sub.node)

if __name__ == '__main__':
    main()
