import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def custom_ocr(video_path, output_video_path, weights_path, digit_model_path):
    def resize_region(region, scale):
        h, w = region.shape[:2]
        if h == 0 or w == 0:
            return None
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(region, (new_w, new_h))

    def adjust_brightness_contrast(image, alpha=1.2, beta=30):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def detect_digits_with_adjustments(digit_model, number_plate_region):
        if number_plate_region is None or number_plate_region.size == 0:
            return []
        
        attempts = [
            lambda img: img if img is not None else None,  # Original
            lambda img: adjust_brightness_contrast(img, 1.5, 50),  # Increase contrast
            lambda img: cv2.GaussianBlur(img, (5, 5), 0) if img is not None else None  # Blur noise reduction
        ]

        for attempt in attempts:
            processed_region = attempt(number_plate_region)
            if processed_region is None or processed_region.size == 0:
                continue
            
            results = digit_model(processed_region, conf=0.1)
            detected_digits = []

            for result in results[0].boxes:
                x1, y1, x2, y2 = result.xyxy[0].tolist()
                conf = result.conf[0].item()
                cls = int(result.cls[0].item())
                label = str(cls)
                detected_digits.append((x1, y1, x2, y2, label))

            if detected_digits:
                return detected_digits

        return []

    model = YOLO(weights_path)
    digit_model = YOLO(digit_model_path)
    tracker = DeepSort(max_age=30)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    stored_digits = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []
        
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            label = model.names[int(result.cls[0].item())]
            conf = result.conf[0].item()

            if label == 'Number Plate':
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'Number Plate'))
        
        tracked_objects = tracker.update_tracks(detections, frame=frame)
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, w, h = track.to_tlwh()
            x2, y2 = x1 + w, y1 + h

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            number_plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
            if number_plate_region.size == 0:
                continue
            
            digits = detect_digits_with_adjustments(digit_model, number_plate_region)
            sorted_digits = sorted(digits, key=lambda d: d[0])
            
            if len(sorted_digits) == 5:
                stored_digits[track_id] = "".join([digit[4] for digit in sorted_digits])
            
            if track_id in stored_digits:
                cv2.putText(frame, f"Digits: {stored_digits[track_id]}", (int(x1), int(y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {output_video_path}")

def main():
    weights_path = '/home/ubuntu/app/weights/21_January_25_weights.pt'
    digit_model_path = '/home/ubuntu/app/weights/digit_detection.pt'
    video_path = '/home/ubuntu/app/videos/Number Plate Identification Improperly parked SUV.mp4'
    output_video_path = '/home/ubuntu/app/output/output_video_with_digits.mp4'

    custom_ocr(video_path, output_video_path, weights_path, digit_model_path)

if __name__ == "__main__":
    main()
