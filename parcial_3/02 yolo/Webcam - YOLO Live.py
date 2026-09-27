import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
ONLY_PERSONS = False  # False para detectar todos los objetos
MODEL_PATH = "parcial_3/yolo/local/weights/yolo11n.pt"
CSV_FILE = "parcial_3/yolo/local/runs/detect/predict/webcam_detections.csv"

def main():
    print(f"Cargando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Asegurar que el directorio del CSV existe
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    
    # Preparar el archivo CSV si no existe
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'ID', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

    print("Iniciando cámara... (Presiona 'q' para salir)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferencia con tracking
            # - persist=True: Mantiene los IDs entre frames
            # - classes=[0]: Filtra solo personas (opcional)
            classes_to_detect = [0] if ONLY_PERSONS else None
            results = model.track(frame, persist=True, classes=classes_to_detect, verbose=False)

            # Obtener el frame con las cajas dibujadas por Ultralytics
            annotated_frame = results[0].plot()

            # Guardar datos de detección en el CSV
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()

                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    for box, obj_id, conf, cls in zip(boxes, ids, confs, clss):
                        writer.writerow([
                            now, int(obj_id), model.names[int(cls)], 
                            round(float(conf), 2), 
                            round(float(box[0]), 1), round(float(box[1]), 1),
                            round(float(box[2]), 1), round(float(box[3]), 1)
                        ])

            # Mostrar la ventana explícitamente
            cv2.imshow("YOLO Deep Learning UAG - Webcam Demo", annotated_frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Cámara cerrada. Detecciones guardadas en: {CSV_FILE}")

if __name__ == "__main__":
    main()
