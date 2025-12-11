import cv2
import time
import multiprocessing
import os
from ultralytics import YOLO

CAMERAS = [
    {"id": "CAM_01_REAL", "source": 0},
    {"id": "CAM_02_ENTRADA", "source": "video_entrada.mp4"},
    {"id": "CAM_03_VIDEO2", "source": "video2.mp4"}
]

def proceso_camara(cam_id, source, queue_salida):
    try:
        model = YOLO('yolov8n.pt') 
    except Exception as e:
        print(f"[{cam_id}] Error: {e}")
        return

    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[{cam_id}] Error al abrir fuente: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    nombre_log = f"registro_{cam_id}.txt"
    nombre_video = f"grabacion_{cam_id}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(nombre_video, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        results = model(frame, conf=0.5, verbose=False)
        
        detecciones = results[0].boxes
        if len(detecciones) > 0:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            mensaje = f"[{timestamp}] Detecciones: {len(detecciones)}\n"
            
            with open(nombre_log, "a") as f:
                f.write(mensaje)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        cv2.putText(annotated_frame, f"CAM: {cam_id}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame_small = cv2.resize(annotated_frame, (640, 360))
        cv2.imshow(f"MONITOR: {cam_id}", frame_small)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyWindow(f"MONITOR: {cam_id}")

def main():
    procesos = []

    for cam in CAMERAS:
        if isinstance(cam["source"], str) and not os.path.exists(cam["source"]):
            print(f"Advertencia: Archivo no encontrado {cam['source']}")
        
        p = multiprocessing.Process(target=proceso_camara, args=(cam["id"], cam["source"], None))
        p.start()
        procesos.append(p)

    for p in procesos:
        p.join()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()