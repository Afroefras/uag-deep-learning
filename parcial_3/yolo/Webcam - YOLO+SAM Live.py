import cv2
import torch
from ultralytics import YOLO, SAM

# --- CONFIGURACIÓN ---
ONLY_PERSONS = False  # False para detectar todos los objetos
YOLO_MODEL_PATH = "parcial_3/yolo/local/weights/yolo11n.pt"
SAM_MODEL_PATH = "parcial_3/yolo/local/weights/mobile_sam.pt"

def main():
    # Detectar si hay GPU disponible para acelerar la inferencia
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    print(f"Cargando modelo YOLO: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print(f"Cargando modelo SAM (MobileSAM): {SAM_MODEL_PATH}")
    sam_model = SAM(SAM_MODEL_PATH)

    print("\nIniciando cámara... (Presiona 'q' para salir)")
    print("Nota: Procesar SAM en tiempo real es exigente, los FPS pueden ser menores que usando solo YOLO.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. YOLO detecta y clasifica
            classes_to_detect = [0] if ONLY_PERSONS else None
            resultados_yolo = yolo_model(frame, classes=classes_to_detect, verbose=False, device=device)[0]
            
            # Si YOLO encontró al menos un objeto
            if len(resultados_yolo.boxes) > 0:
                cajas_yolo = resultados_yolo.boxes.xyxy 
                
                # 2. SAM segmenta usando las cajas de YOLO
                resultados_sam = sam_model(
                    source=frame,
                    bboxes=cajas_yolo,
                    device=device,
                    verbose=False,
                    retina_masks=False # False para priorizar velocidad en video en vivo
                )[0]
                
                # 3. Trasladar clases de YOLO a SAM (Pipeline "El Matrimonio Perfecto")
                # Copiamos los nombres de las clases
                resultados_sam.names = resultados_yolo.names 
                # Clonamos el tensor para poder editarlo
                nuevo_tensor = resultados_sam.boxes.data.clone()
                # Asignamos las clases identificadas por YOLO en la columna correspondiente de SAM
                nuevo_tensor[:, 5] = resultados_yolo.boxes.cls
                resultados_sam.boxes.data = nuevo_tensor
                
                # Obtener el frame con las máscaras y cajas dibujadas con las etiquetas de YOLO
                annotated_frame = resultados_sam.plot(boxes=True)
            else:
                # Si no hay detecciones, mostramos el frame original
                annotated_frame = frame

            # Mostrar la ventana explícitamente
            cv2.imshow("YOLO + SAM Deep Learning UAG - Webcam Demo", annotated_frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cámara cerrada.")

if __name__ == "__main__":
    main()
