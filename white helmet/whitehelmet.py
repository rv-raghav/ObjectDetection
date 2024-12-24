import cv2
import os
import numpy as np
import math
from ultralytics import YOLO


# Function to process videos and extract frames with white helmets
def video_detection(video_path, output_frames_folder):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    model = YOLO("../YOLO-Weights/yolov8n.pt")
    classNames = ["person", "helmet"]  # Adjusted to include only "person" and "helmet" classes

    frame_count = 0
    white_helmet_detected = False

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_count += 1
        results = model(img, stream=True)

        persons = []
        helmets = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if cls < len(classNames):
                    class_name = classNames[cls]

                    if class_name == "person" and conf > 0.5:  # Increase confidence threshold if necessary
                        persons.append((x1, y1, x2, y2, conf))
                    elif class_name == "helmet" and conf > 0.5:  # Increase confidence threshold if necessary
                        helmets.append((x1, y1, x2, y2, conf))

        for (px1, py1, px2, py2, pconf) in persons:
            person_roi = img[py1:py2, px1:px2]
            person_height = py2 - py1

            roi_top = max(py1 - int(person_height * 0.3), 0)
            roi_bottom = py2

            head_roi = img[roi_top:roi_bottom, px1:px2]
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)

            if cv2.countNonZero(white_mask) > 0:
                white_helmet_detected = True
                frame_output_path = os.path.join(output_frames_folder, f"{video_name}_frame{frame_count}.jpg")
                cv2.imwrite(frame_output_path, img)
                break

        if white_helmet_detected:
            break

    cap.release()


# Main function to process all videos in a folder
def process_videos_in_folder(input_folder, output_frames_folder):
    os.makedirs(output_frames_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".avi", ".mov"))]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_detection(video_path, output_frames_folder)
        print(f"Processed {video_file}")


# Specify input folder and output folder paths
input_folder = r"D:\Person not wearing helmet"
output_frames_folder = r"C:\Users\ragha\Downloads\output_folder"

# Process all videos in the input folder
process_videos_in_folder(input_folder, output_frames_folder)
