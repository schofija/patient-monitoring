import cv2
import numpy as np

# Load the model
weights = './yolov7-tiny.weights'
config = './yolov7-tiny.cfg'
model = cv2.dnn_DetectionModel(config, weights)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect objects
    classes, confidences, boxes = model.detect(frame, confThreshold=0.2, nmsThreshold=0.4)

    # Draw bounding boxes
    for class_id, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('YOLOv7 Webcam', frame)
    key = cv2.waitKey(1)
    if key == 27:  #Break using ESC
        break

cap.release()
cv2.destroyAllWindows()
