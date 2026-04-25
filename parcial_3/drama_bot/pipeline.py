import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import ollama # Asumiendo que usamos Ollama para Gemma 4 local

class DramaPipeline:
    def __init__(self):
        # Carga de modelos (Usando versiones ligeras para el MVP)
        print("Cargando YOLOv8n...")
        self.yolo = YOLO('yolov8n.pt') 
        
        print("Cargando SAM...")
        self.sam = SAM('sam_b.pt') # O el modelo que prefieras
        
    def process_image(self, image_path):
        """
        Flujo: YOLO (Detección) -> SAM (Segmentación) -> Gemma (Razonamiento)
        """
        # 1. Detección de Personas/Caras con YOLO
        results = self.yolo(image_path, classes=[0]) # Clase 0 es persona
        if not results or len(results[0].boxes) == 0:
            return "No encontré a ningún humano para dramatizar. ¿Eres un bot?", None

        # Tomamos la detección más confiable
        bbox = results[0].boxes.xyxy[0].cpu().numpy()
        
        # 2. Segmentación con SAM usando el BBox de YOLO como prompt
        # Esto es lo que les volará la cabeza: SAM no adivina, YOLO le dice dónde ver
        sam_results = self.sam(image_path, bboxes=[bbox])
        
        # 3. Preparar el crop para Gemma
        img = cv2.imread(image_path)
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = img[y1:y2, x1:x2]
        crop_path = "temp_crop.jpg"
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
        
        response = ollama.generate(
            model='gemma:4b', # O el nombre exacto de tu modelo Gemma 4
            prompt=prompt,
            images=[crop_path]
        )
        
        return response['response'], crop_path

# Prueba rápida si se ejecuta solo
if __name__ == "__main__":
    # pipeline = DramaPipeline()
    # print(pipeline.process_image('test.jpg'))
    pass
