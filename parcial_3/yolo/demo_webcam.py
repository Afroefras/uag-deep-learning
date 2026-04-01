import cv2
from ultralytics import YOLO

def main():
    print("Iniciando cámara térmica (presiona 'q' para salir)...")
    
    # Cargar el modelo preentrenado (nano, perfecto para CPU y tiempo real)
    model = YOLO("yolo11n.pt")
    
    # Inferencia continua utilizando la webcam (source=0)
    # Parametros:
    # - show=True: Abre una ventana interactiva
    # - conf=0.4: Filtra detecciones con confianza menor al 40%
    # - classes: Aquí podríamos usar [0] p. ej. para detectar solo personas
    model.predict(source=0, show=True, conf=0.4)

if __name__ == "__main__":
    main()
