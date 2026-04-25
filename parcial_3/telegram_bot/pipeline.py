import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO, SAM
from google import genai
import ollama

class DramaPipeline:
    def __init__(self):
        """
        Constructor: Inicializa modelos y clientes.
        Diseñado para ser agnóstico: carga su propio entorno y no depende de carpetas externas.
        """
        # 1. Cargar variables de entorno (API Keys)
        load_dotenv()
        
        # 2. Configuración de rutas (Todo bajo la carpeta weights por orden)
        self.weights_dir = Path(r"parcial_3\telegram_bot\weights")
        self.output_dir = self.weights_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Carga de modelos de Visión (YOLO + SAM)
        print("Cargando YOLOv8n...")
        self.yolo = YOLO(str(self.weights_dir / "yolo11n.pt")) 
        
        print("Cargando SAM...")
        self.sam = SAM(str(self.weights_dir / "mobile_sam.pt"))

        # 4. Inicializar cliente de Google GenAI para Imagen (El "NanoBanana")
        # Usamos el SDK directamente para que los estudiantes vean la lógica simple.
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
            print("Cliente Gemini Imagen listo! 🎨")
        else:
            self.gemini_client = None
            print("⚠️ Aviso: GOOGLE_API_KEY no encontrada en .env. La generación de imágenes estará desactivada.")
        
    def process_image(self, image_path):
        """
        Flujo completo: YOLO -> SAM -> Gemma -> Imagen
        """
        # --- PASO 1: DETECCIÓN (YOLO) ---
        results = self.yolo(image_path, classes=[0]) # Clase 0 = persona
        if not results or len(results[0].boxes) == 0:
            return "No encontré a ningún humano. Acaso eres un bot?", None

        bbox = results[0].boxes.xyxy[0].cpu().numpy() # Tomamos la detección más confiable
        
        # --- PASO 2: SEGMENTACIÓN (SAM) ---
        sam_results = self.sam(image_path, bboxes=[bbox], verbose=False)
        
        # --- PASO 3: RECORTE Y MÁSCARA (OpenCV) ---
        img = cv2.imread(image_path)
        if img is None: return "Error al leer imagen.", None
        
        if sam_results[0].masks is None:
            return "No pude segmentar tu carita preciosa", None
            
        # Convertimos la máscara de bool a uint8 para que OpenCV pueda procesarla
        mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Aplicamos la máscara (multiplicación por 0 o 1)
        masked_img = img * mask[:, :, np.newaxis]
        
        # Hacemos el crop (recorte) final
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = masked_img[y1:y2, x1:x2]
        
        crop_path = str(self.output_dir / "temp_crop.jpg")
        cv2.imwrite(crop_path, face_crop)

        # --- PASO 4: RAZONAMIENTO Y PROMPT (Gemma vía Ollama) ---
        prompt_gemma = """
        Analiza esta expresión facial. 
        1. Inventa una biografía trágica y exagerada de 2 enunciados sobre por qué esta persona está así.
        2. Genera un prompt de 10 palabras para crear una caricatura de esta persona en estilo 'Nano Banana Digital Art'.
           El prompt debe pedir rasgos extremadamente exagerados y dramáticos para un efecto cómico de caricatura.
        
        Responde estrictamente en este formato:
        HISTORIA: [Tu historia]
        PROMPT: [Tu prompt]
        """
        
        response = ollama.generate(model="gemma4:e4b", prompt=prompt_gemma, images=[crop_path])
        raw_text = response['response']
        
        # --- PASO 5: GENERACIÓN ARTÍSTICA (Gemini Imagen) ---
        final_image_path = crop_path # Por defecto usamos el recorte si falla Imagen
        
        if self.gemini_client and "PROMPT:" in raw_text:
            try:
                # Extraemos el prompt artístico generado por Gemma
                artist_prompt = raw_text.split("PROMPT:")[1].strip()
                print(f"Generando Caricatura: {artist_prompt}")
                
                img_response = self.gemini_client.models.generate_image(
                    model='gemini-2.5-flash-image',
                    prompt=artist_prompt,
                    config={'number_of_images': 1}
                )
                
                caricature_path = str(self.output_dir / "caricature.jpg")
                img_response.generated_images[0].image.save(caricature_path)
                final_image_path = caricature_path
                print("¡NanoBanana generado con éxito!")
            except Exception as e:
                print(f"Error en Imagen: {e}")
        
        return raw_text, final_image_path

# Prueba rápida si se ejecuta solo
if __name__ == "__main__":
    # pipeline = DramaPipeline()
    # print(pipeline.process_image('test.jpg'))
    pass
