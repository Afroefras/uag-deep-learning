# 🎭 Telegram Bot MVP

Este es un ejemplo de **Pipeline de Integración Multimodelo** diseñado para demostrar cómo se conectan las piezas de Visión Artificial y Razonamiento (LLM) en un producto real.

## Arquitectura del Proyecto
1. **Telegram Bot (`app.py`)**: La interfaz de usuario. Recibe la imagen y entrega el resultado final.
2. **YOLO (`pipeline.py`)**: El "Buscador". Su única tarea es encontrar una cara en la imagen y darnos coordenadas (BBox).
3. **SAM (`pipeline.py`)**: El "Recortador de Precisión". Usa el BBox de YOLO para segmentar la cara con exactitud.
4. **Gemma 4 (`ollama`)**: El "Cerebro". Recibe el recorte de la imagen y genera una narrativa basada en la expresión detectada.

## Flujo de Datos
`Foto del Alumno` -> `YOLO (Persona)` -> `SAM (Segmento)` -> `Crop` -> `Gemma (Historia + Prompt)` -> `Respuesta al Alumno`

## Cómo usarlo en clase
1. Corre `python app.py`.
2. Tómate una selfie con cara de estrés.
3. Explica a los alumnos:
   - "YOLO ya encontró mi cara (Detección)".
   - "SAM está eliminando el fondo para que Gemma no se distraiga (Segmentación)".
   - "Gemma está razonando por qué tengo esta cara (Inferencia)".

## Dependencias
```bash
pip install python-telegram-bot ultralytics opencv-python ollama python-dotenv
```

> **Seguridad:** El token de Telegram debe ir en el archivo `.env`. Nunca subas ese archivo a repositorios públicos.
