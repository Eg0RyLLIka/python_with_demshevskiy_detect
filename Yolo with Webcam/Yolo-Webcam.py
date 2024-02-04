from ultralytics import YOLO
import cv2
import cvzone
import math

# Cap for the Webcam
# cap = cv2.VideoCapture(1) # For Webcam or 0  (если нет то, opencv-python смотреть версию 4.5.4.60 and etc +)
# cap.set(3, 1280)
# cap.set(4, 720)

# Cap for the Video

cap = cv2.VideoCapture("../Videos/cats.mp4") # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")  # мал/бол - n/l

classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
'teddy bear', 'hair drier', 'toothbrush']

while True:

	success, img = cap.read()
	results = model(img, stream=True)

	for r in results:

		boxes = r.boxes

		for box in boxes:

			# x1, y1, x2, y2 = box.xyxy[0]

			# x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

			# print(x1, y1, x2, y2)

			# cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) #3 el - colour, 4 el - thickness

			# x1, y1, w, h = box.xywh[0]

			# bbox = int(x1), int(y1), int(w), int(h)

			# print(x1, y1, w, h)

			# cvzone.cornerRect(img, bbox)

			# Bounding Box

			x1, y1, x2, y2 = box.xyxy[0]

			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

			w, h = x2 - x1, y2 - y1

			cvzone.cornerRect(img, (x1, y1, w, h))  # - посмотреть в функкции аргументы - цвет и т.п

			# Confidence

			conf = math.ceil((box.conf[0] * 100)) / 100 # - rounded # - возможно лишние скобки

			# cvzone.putTextRect(img, f'{conf}, (max(0, x1), max(35, y1))) - снизу дополнен

			# Class Name

			cls = int(box.cls[0])

			cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

	cv2.imshow("Image", img)

	cv2.waitKey(1)

	# results = model("Images/1.png", show=True)

	# cv2.waitKey(0)


#	- directory of project(Videos)

#	- найти видео рандомные