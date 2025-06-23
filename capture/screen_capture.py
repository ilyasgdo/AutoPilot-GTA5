import cv2
import numpy as np
import mss
from perception.detector import YoloDetector
from perception.lane_detection import detect_lanes


# Zone de capture (adapte à ta résolution ou à GTA en fenêtré)
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

def get_frame(resize_shape=(224, 224)):
    with mss.mss() as sct:
        frame = np.array(sct.grab(monitor))[:, :, :3]  # Supprime l'alpha
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize_shape)
        return frame

detector = YoloDetector()

if __name__ == "__main__":
    while True:
        frame = get_frame()
        
        # Détection de lignes
        lane_frame, _ = detect_lanes(frame)

        # Détection objets
        results = detector.detect(lane_frame)
        full_frame = detector.draw(lane_frame, results)

        cv2.imshow("YOLO + Lanes", full_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
