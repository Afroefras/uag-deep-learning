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
        Diseñado para ser híbrido: Vertex AI para Inpainting PRO o Developer API para modo clase.
        """
        # 1. Cargar variables de entorno
        load_dotenv()
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")
        self.location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        # 2. Configuración de rutas
        self.weights_dir = Path(r"parcial_3\telegram_bot\weights")
        self.output_dir = self.weights_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Carga de modelos de Visión
        print("Cargando YOLOv8n...")
        self.yolo = YOLO(str(self.weights_dir / "yolo11n.pt")) 
        print("Cargando SAM...")
        self.sam = SAM(str(self.weights_dir / "mobile_sam.pt"))

        # 4. Inicialización de Cliente Híbrido (Vertex vs Gemini)
        self.gemini_client = None
        self.mode = None

        if self.project_id:
            try:
                # Modo PRO: Vertex AI (Soporta Inpainting/Edit)
                self.gemini_client = genai.Client(
                    vertexai=True, 
                    project=self.project_id, 
                    location=self.location
                )
                self.mode = "VERTEX"
                print(f"🔥 Modo PRO activado: Vertex AI en {self.project_id}")
            except Exception as e:
                print(f"Error inicializando Vertex AI: {e}. Intentando fallback...")
                self.project_id = None

        if not self.project_id:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                # Modo Clase: Developer API (Soporta Generate Images estándar)
                self.gemini_client = genai.Client(api_key=api_key)
                self.mode = "DEVELOPER"
                print("🚀 Modo Clase: Usando Developer API Key (Inpainting no soportado)")
            else:
                print("⚠️ Aviso: No se encontró API Key ni Project ID en .env")

    def process_image(self, image_path):
        """
        Flujo completo: YOLO -> SAM -> Gemma -> Imagen (Híbrido)
        """
        # --- PASO 1: DETECCIÓN (YOLO) ---
        results = self.yolo(image_path, classes=[0])
        if not results or len(results[0].boxes) == 0:
            return "No encontré a ningún humano. Acaso eres un bot?", None

        bbox = results[0].boxes.xyxy[0].cpu().numpy()
        
        # --- PASO 2: SEGMENTACIÓN (SAM) ---
        sam_results = self.sam(image_path, bboxes=np.array([bbox]), verbose=False)
        
        # --- PASO 3: RECORTE Y MÁSCARA (OpenCV) ---
        img = cv2.imread(image_path)
        if img is None: return "Error al leer imagen.", None
        
        if sam_results[0].masks is None:
            return "No pude segmentar tu carita preciosa", None
            
        mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Preparar crop para Gemma
        masked_img = img * mask[:, :, np.newaxis]
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = masked_img[y1:y2, x1:x2]
        crop_path = str(self.output_dir / "temp_crop.jpg")
        cv2.imwrite(crop_path, face_crop)
        
        # Preparar imágenes para Inpainting (Vertex)
        mask_inpainting = mask * 255
        _, base_img_encoded = cv2.imencode('.jpg', img)
        base_image_bytes = base_img_encoded.tobytes()
        _, mask_img_encoded = cv2.imencode('.png', mask_inpainting)
        mask_image_bytes = mask_img_encoded.tobytes()

        # --- PASO 4: RAZONAMIENTO Y PROMPT (Gemma vía Ollama) ---
        prompt_gemma = """
        Analiza esta expresión facial y el ambiente. 
        1. Inventa una biografía profundamente dramática y SARCÁSTICA sobre por qué esta persona está así.
           ENFÓCATE en el agotamiento existencial de ser un estudiante de IA que debe aprender algo nuevo cada día porque todo se vuelve obsoleto en 24 horas.
           HÁBLALE DIRECTAMENTE AL USUARIO (en segunda persona, ej: 'Tu mirada grita que...').
           Sé breve y contundente (máximo 2 enunciados).
        2. Genera un prompt de 10 palabras para transformar ESTA FIGURA en una caricatura estilo 'Nano Banana Digital Art'.
           Describe ÚNICAMENTE cómo alterar los rasgos humanos (ej: exagerar ojos, añadir lentes dramáticos, bigote de poeta).
           No menciones el fondo.
        
        Responde estrictamente en este formato:
        HISTORIA: [Tu historia]
        PROMPT: [Tu prompt]
        """
        
        response = ollama.generate(model="gemma4:e4b", prompt=prompt_gemma, images=[crop_path])
        raw_text = response['response']
        
        # --- PASO 5: GENERACIÓN ARTÍSTICA (Vertex Inpainting vs Developer standard) ---
        final_image_path = crop_path
        if self.gemini_client and "PROMPT:" in raw_text:
            try:
                from google.genai import types
                artist_prompt = raw_text.split("PROMPT:")[1].strip()
                
                if self.mode == "VERTEX":
                    # INPAINTING REAL (Solo Vertex AI)
                    print(f"Generando Inpainting en Vertex: {artist_prompt}")
                    raw_ref = types.RawReferenceImage(
                        reference_id=1,
                        reference_image=types.Image(image_bytes=base_image_bytes, mime_type="image/jpeg")
                    )
                    mask_ref = types.MaskReferenceImage(
                        reference_id=2,
                        reference_image=types.Image(image_bytes=mask_image_bytes, mime_type="image/png"),
                        config=types.MaskReferenceConfig(mask_mode="MASK_MODE_USER_PROVIDED")
                    )
                    img_response = self.gemini_client.models.edit_image(
                        model='imagen-4.0-fast-generate-001',
                        prompt=artist_prompt,
                        reference_images=[raw_ref, mask_ref],
                        config=types.EditImageConfig(edit_mode="EDIT_MODE_INPAINT_INSERTION", number_of_images=1)
                    )
                else:
                    # TEXT-TO-IMAGE FALLBACK (Developer API)
                    print(f"Generando Caricatura Estándar: {artist_prompt}")
                    img_response = self.gemini_client.models.generate_images(
                        model='imagen-4.0-fast-generate-001',
                        prompt=artist_prompt,
                        config={'number_of_images': 1}
                    )
                
                caricature_path = str(self.output_dir / "caricature.jpg")
                img_response.generated_images[0].image.save(caricature_path)
                final_image_path = caricature_path
                print(f"¡NanoBanana ({self.mode}) generado!")
            except Exception as e:
                print(f"Error en Generación de Imagen: {e}")
        
        user_text = raw_text.split("PROMPT:")[0].replace("HISTORIA:", "").strip()
        return user_text, final_image_path


# Prueba rápida si se ejecuta solo
if __name__ == "__main__":
    # pipeline = DramaPipeline()
    # print(pipeline.process_image('test.jpg'))
    pass
