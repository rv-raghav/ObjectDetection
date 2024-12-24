from ultralytics import YOLO
import cv2
import numpy as np
import math

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    model = YOLO("../YOLO-Weights/yolov8n.pt")
    classNames = ["person", "helmet"]  # Adjusted to include only "person" and "helmet" classes

    while True:
        success, img = cap.read()
        if not success:
            break

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

                # Process only "person" and "helmet" classes
                if cls < len(classNames):
                    class_name = classNames[cls]

                    if class_name == "person" and conf > 0.5:  # Increase confidence threshold if necessary
                        persons.append((x1, y1, x2, y2, conf))
                    elif class_name == "helmet" and conf > 0.5:  # Increase confidence threshold if necessary
                        helmets.append((x1, y1, x2, y2, conf))

        # Loop through detected persons and check for helmets
        for (px1, py1, px2, py2, pconf) in persons:
            person_roi = img[py1:py2, px1:px2]
            person_height = py2 - py1

            # Expand ROI vertically
            roi_top = max(py1 - int(person_height * 0.3), 0)
            roi_bottom = py2

            head_roi = img[roi_top:roi_bottom, px1:px2]

            # Convert to HSV for color detection
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

            # Define color ranges for detection (yellow, blue, white)
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            blue_lower = np.array([100, 150, 0])
            blue_upper = np.array([140, 255, 255])
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])

            # Create masks for each color range
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)

            # Combine masks to detect any of the three colors
            combined_mask = cv2.bitwise_or(yellow_mask, cv2.bitwise_or(blue_mask, white_mask))

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if significant colored area is detected
            helmet_detected = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 150:  # Adjust this threshold as needed
                    helmet_detected = True
                    break

            # Draw bounding box for person and helmet status
            color = (0, 255, 0) if helmet_detected else (0, 0, 255)
            cv2.rectangle(img, (px1, roi_top), (px2, roi_bottom), color, 3)

            label = f'Person {"with helmet" if helmet_detected else "without helmet"} {pconf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = px1 + t_size[0], roi_top - t_size[1] - 3

            cv2.rectangle(img, (px1, roi_top), c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (px1, roi_top - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Fire extinguisher detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])
        red_upper2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # Check if the contour area is significant and if the aspect ratio is typical for a fire extinguisher
            if area > 200 and 0.2 < aspect_ratio < 0.6:  # Adjust thresholds as needed
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)  # Yellow color for fire extinguisher
                label = 'Fire Extinguisher'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x + t_size[0], y - t_size[1] - 3
                cv2.rectangle(img, (x, y), c2, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(img, label, (x, y - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        out.write(img)
        yield img

    cap.release()
    out.release()
    cv2.destroyAllWindows()