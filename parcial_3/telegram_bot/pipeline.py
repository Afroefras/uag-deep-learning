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
        sam_results = self.sam(image_path, bboxes=np.array([bbox]), verbose=False)
        
        # --- PASO 3: RECORTE Y MÁSCARA (OpenCV) ---
        img = cv2.imread(image_path)
        if img is None: return "Error al leer imagen.", None
        
        if sam_results[0].masks is None:
            return "No pude segmentar tu carita preciosa", None
            
        # Convertimos la máscara de bool a uint8 para que OpenCV pueda procesarla
        mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Aplicamos la máscara (multiplicación por 0 o 1) para el crop
        masked_img = img * mask[:, :, np.newaxis]
        
        # Hacemos el crop (recorte) final para Gemma
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = masked_img[y1:y2, x1:x2]
        
        crop_path = str(self.output_dir / "temp_crop.jpg")
        cv2.imwrite(crop_path, face_crop)
        
        # --- PREPARACIÓN PARA INPAINTING ---
        # Máscara binaria: 255 (Blanco) = Área a editar (Persona), 0 (Negro) = Área a preservar (Fondo)
        mask_inpainting = mask * 255
        
        _, base_img_encoded = cv2.imencode('.jpg', img)
        base_image_bytes = base_img_encoded.tobytes()
        
        _, mask_img_encoded = cv2.imencode('.png', mask_inpainting)
        mask_image_bytes = mask_img_encoded.tobytes()

        # --- PASO 4: RAZONAMIENTO Y PROMPT (Gemma vía Ollama) ---
        prompt_gemma = """
        Analiza esta expresión facial y el ambiente. 
        1. Inventa una biografía profundamente dramática, existencial y exagerada sobre por qué esta persona está así.
           HÁBLALE DIRECTAMENTE AL USUARIO (en segunda persona, ej: 'Tu mirada revela que...').
           Sé breve y contundente (máximo 2 enunciados). Evita modismos modernos o slang.
        2. Genera un prompt de 10 palabras para transformar ESTA FIGURA en una caricatura estilo 'Nano Banana Digital Art'.
           Describe ÚNICAMENTE cómo alterar los rasgos humanos (ej: exagerar ojos, añadir lentes dramáticos, bigote de poeta).
           No menciones el fondo, ya que será preservado mediante una máscara.
        
        Responde estrictamente en este formato:
        HISTORIA: [Tu historia]
        PROMPT: [Tu prompt]
        """
        
        response = ollama.generate(model="gemma4:e4b", prompt=prompt_gemma, images=[crop_path])
        raw_text = response['response']
        
        # --- PASO 5: GENERACIÓN ARTÍSTICA (Gemini Imagen - Inpainting) ---
        final_image_path = crop_path # Por defecto usamos el recorte si falla Imagen
        
        if self.gemini_client and "PROMPT:" in raw_text:
            try:
                from google.genai import types
                
                # Extraemos el prompt artístico generado por Gemma
                artist_prompt = raw_text.split("PROMPT:")[1].strip()
                print(f"Generando Inpainting: {artist_prompt}")
                
                # Configurar objetos de imagen con data/mime_type
                image_input = types.Image(image_bytes=base_image_bytes, mime_type="image/jpeg")
                mask_input = types.Image(image_bytes=mask_image_bytes, mime_type="image/png")
                
                # Llamada a Imagen 4.0 Fast con Inpainting
                img_response = self.gemini_client.models.generate_images(
                    model='imagen-4.0-fast-generate-001',
                    prompt=artist_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        masked_image=types.MaskedImage(
                            image=image_input,
                            mask=mask_input,
                            mask_mode="MASK_MODE_INPAINT_ADDITION"
                        )
                    )
                )
                
                caricature_path = str(self.output_dir / "caricature.jpg")
                img_response.generated_images[0].image.save(caricature_path)
                final_image_path = caricature_path
                print("¡NanoBanana (Inpainting) generado con éxito!")
            except Exception as e:
                print(f"Error en Inpainting: {e}")
        
        # Limpiamos la respuesta para el usuario (le quitamos el prompt interno)
        user_text = raw_text.split("PROMPT:")[0].replace("HISTORIA:", "").strip()
        return user_text, final_image_path

# Prueba rápida si se ejecuta solo
if __name__ == "__main__":
    # pipeline = DramaPipeline()
    # print(pipeline.process_image('test.jpg'))
    pass
