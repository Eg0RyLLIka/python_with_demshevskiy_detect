from ultralytics import YOLO

import cv2

model = YOLO('../Yolo-Weights/yolov8l.pt')    # - n or l(large)

results = model("Images/1.png", show=True)  # - 1 or 2 or 3

cv2.waitKey(0)