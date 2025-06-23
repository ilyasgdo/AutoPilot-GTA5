from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.model.fuse()  # Pour accélérer l'inférence
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)[0]
        return results

    def draw(self, frame, results):
        annotated = frame.copy()
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return annotated
