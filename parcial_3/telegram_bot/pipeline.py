import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import ollama # Asumiendo que usamos Ollama para Gemma 4 local

class DramaPipeline:
    def __init__(self):
        # Carga de modelos (Usando versiones ligeras para el MVP)
        print("Cargando YOLOv8n...")
        YOLO_PATH = r"weights\yolo11n.pt"
        self.yolo = YOLO(YOLO_PATH) 
        
        print("Cargando SAM...")
        SAM_PATH = r"weights\sam_b.pt"
        self.sam = SAM(SAM_PATH)
        
    def process_image(self, image_path):
        """
        Flujo: YOLO (Detección) -> SAM (Segmentación) -> Gemma (Razonamiento)
        """
        # 1. Detección de Personas/Caras con YOLO
        results = self.yolo(image_path, classes=[0]) # Clase 0 es persona
        if not results or len(results[0].boxes) == 0:
            return "No encontré a ningún humano. Acaso eres un bot?", None

        # Tomamos la detección más confiable
        bbox = results[0].boxes.xyxy[0].cpu().numpy()
        
        # 2. Segmentación con SAM usando el BBox de YOLO como prompt
        sam_results = self.sam(image_path, bboxes=[bbox], verbose=False)
        
        # 3. Aplicar Máscara de SAM y preparar el crop
        img = cv2.imread(image_path)
        
        # Extraemos la máscara binaria (0 y 1)
        # La redimensionamos al tamaño original de la imagen si es necesario
        mask = sam_results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        # Multiplicamos la imagen por la máscara para dejar el fondo en negro
        # Convertimos la máscara a 3 canales para que coincida con BGR
        masked_img = img * mask[:, :, np.newaxis].astype(np.uint8)
        
        # Ahora sí, hacemos el crop sobre la imagen segmentada
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = masked_img[y1:y2, x1:x2]
        
        crop_path = "assets/images/faces/temp_crop.jpg"
        cv2.imwrite(crop_path, face_crop)

        # 4. Razonamiento con Gemma 4
        # Le pedimos la historia trágica y el prompt para la caricatura
        prompt = """
        Analiza esta expresión facial. 
        1. Inventa una biografía trágica y exagerada de 2 enunciados sobre por qué esta persona está así.
        2. Genera un prompt de 10 palabras para crear una caricatura de esta persona en estilo 'Nano Banana Digital Art'.
        
        Responde en este formato:
        HISTORIA: [Tu historia]
        PROMPT: [Tu prompt]
        """
        
        MODEL = "gemma4:e4b"
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            images=[crop_path]
        )
        
        return response['response'], crop_path

# Prueba rápida si se ejecuta solo
if __name__ == "__main__":
    # pipeline = DramaPipeline()
    # print(pipeline.process_image('test.jpg'))
    pass
