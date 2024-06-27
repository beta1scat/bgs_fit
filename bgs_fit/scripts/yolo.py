import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO("../../data/models/yolov8n.pt")
image_path = '/root/ros_ws/src/data/image.png'
# image_path = '/root/ros_ws/src/data/table.jpg'
im2 = cv2.imread(image_path)
results = model.predict(source=im2, save=True, save_txt=True)
