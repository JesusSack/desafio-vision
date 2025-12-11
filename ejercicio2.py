import cv2
import threading
import queue
import time
import json
from datetime import datetime
from ultralytics import YOLO

VIDEO_SOURCE = 0
CONF_THRESHOLD = 0.5
MODEL_FILE = 'yolov8n-pose.pt'
OUTPUT_VIDEO = "video_procesado.mp4"
OUTPUT_JSON = "datos_telemetria.json"

class RTSPStreamLoader:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.q = queue.Queue()
        self.active = False
        self.thread_stop = False
        
        if self.capture.isOpened():
            self.active = True
            
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.thread_stop:
            if not self.active:
                break
            
            ret, frame = self.capture.read()
            if not ret:
                self.active = False
                break
            
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get(timeout=1)
        except queue.Empty:
            return None

    def running(self):
        return self.active

    def stop(self):
        self.thread_stop = True
        self.thread.join()
        self.capture.release()

def main():
    model = YOLO(MODEL_FILE)
    stream = RTSPStreamLoader(VIDEO_SOURCE)
    
    time.sleep(1.5)
    
    if not stream.running():
        return

    temp_frame = stream.read()
    while temp_frame is None:
        temp_frame = stream.read()
    
    h, w, _ = temp_frame.shape
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
    
    telemetry_data = []

    try:
        while stream.running():
            frame = stream.read()
            if frame is None: continue

            results = model(frame, verbose=False, conf=CONF_THRESHOLD)
            
            timestamp = datetime.now().isoformat()
            frame_log = {
                "ts": timestamp,
                "objects": []
            }

            if results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints else []
                
                for idx, box in enumerate(boxes):
                    obj_data = {
                        "bbox": box.tolist(),
                        "skeleton": keypoints[idx].tolist() if len(keypoints) > idx else []
                    }
                    frame_log["objects"].append(obj_data)

            telemetry_data.append(frame_log)

            annotated_frame = results[0].plot()
            writer.write(annotated_frame)
            cv2.imshow("RTSP Monitor", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        writer.release()
        cv2.destroyAllWindows()
        
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(telemetry_data, f)

if __name__ == "__main__":
    main()