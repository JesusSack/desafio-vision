import cv2
import numpy as np
from ultralytics import YOLO
import threading
import json
from datetime import datetime

VIDEO_SOURCE = "video_entrada.mp4"
OUTPUT_VIDEO = "video_final_procesado.avi"

model = YOLO('yolov8n-pose.pt')

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def upload_s3_simulation(filename, data):
    def _sim_upload():
        json_str = json.dumps(data)
        size_kb = len(json_str) / 1024
        print(f"[SIMULACION S3] Archivo {filename} subido exitosamente ({size_kb:.2f} KB)")

    threading.Thread(target=_sim_upload, daemon=True).start()

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    frame_count = 0
    buffer_data = []
    motion_history = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        results = model.track(frame, persist=True, verbose=False)
        timestamp = datetime.now().isoformat()
        
        current_frame_data = {
            "frame": frame_count, 
            "timestamp": timestamp, 
            "detections": []
        }

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box, track_id, kpts, conf in zip(boxes, track_ids, keypoints, confs):
                if track_id not in motion_history: 
                    motion_history[track_id] = []
                
                if len(kpts) > 0:
                    motion_history[track_id].append(kpts[0][1])

                if len(motion_history[track_id]) > (fps * 9):
                    motion_history[track_id].pop(0)

                status = "ANALYZING"
                if len(motion_history[track_id]) >= fps:
                    std_dev = np.std(motion_history[track_id])
                    if std_dev > 15: status = "ACTIVE"
                    elif std_dev > 5: status = "NORMAL"
                    else: status = "STATIC"

                elbow_angle = 0.0
                arm_side = "None"
                
                if conf is not None:
                    conf_left = conf[5] + conf[7] + conf[9]
                    conf_right = conf[6] + conf[8] + conf[10]
                    
                    if conf_right > conf_left and conf[6] > 0.5 and conf[8] > 0.5 and conf[10] > 0.5:
                        elbow_angle = calculate_angle(kpts[6], kpts[8], kpts[10])
                        arm_side = "R"
                    elif conf_left > conf_right and conf[5] > 0.5 and conf[7] > 0.5 and conf[9] > 0.5:
                        elbow_angle = calculate_angle(kpts[5], kpts[7], kpts[9])
                        arm_side = "L"

                color = (0, 0, 255) if status == "ACTIVE" else (0, 255, 0)
                label = f"ID:{track_id} {status} {arm_side}:{int(elbow_angle)}"
                
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, label, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                current_frame_data["detections"].append({
                    "id": track_id,
                    "status": status,
                    "angle": float(elbow_angle),
                    "arm": arm_side,
                    "bbox": [float(b) for b in box]
                })

        buffer_data.append(current_frame_data)

        if frame_count % fps == 0:
            filename = f"telemetry_{timestamp.replace(':','-')}.json"
            upload_s3_simulation(filename, list(buffer_data))
            buffer_data = []

        out.write(frame)
        cv2.imshow("Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()