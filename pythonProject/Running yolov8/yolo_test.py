from ultralytics import YOLO
import cv2

model=YOLO('yolov8n.pt')
results=model(r"C:\Users\ragha\Downloads\output_folder\frame_0.jpg", show=True)

cv2.waitKey(0)