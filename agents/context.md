# Context for AI Agents: Deep Learning & Production Notebooks (Parcial 3)

This repository contains Jupyter Notebooks used for teaching Deep Learning to last-semester undergraduate students (actuaries/data scientists). When creating new notebooks or modifying existing ones, future AI agents **must** strictly adhere to the following pedagogical and styling guidelines:

## 1. Content and Frameworks (Training vs. Inference)
- **Training**: When building or fine-tuning models from scratch (e.g., CNNs, ResNet, EfficientNet, simple Transformers), strictly use **PyTorch Lightning** (`pytorch_lightning as pl`). Avoid raw PyTorch training loops.
- **Inference & State-of-the-Art**: When teaching inference with modern tools (YOLO, SAM, Hugging Face, LLMs), **DO NOT** wrap them in PyTorch Lightning. Use their native APIs and pipelines:
  - `ultralytics` for YOLO/SAM
  - `transformers` pipelines for Hugging Face (tokenizers, models, attention viz)
  - `ollama` (Python client) for local LLMs — backed by [Ollama](https://ollama.com), which handles CUDA/Metal automatically and supports Gemma 4 natively multimodal
  - `google-genai` for Gemini Embeddings (`gemini-embedding-2-preview`)
- **Target Audience**: Last-semester students with strong mathematical backgrounds but transitioning into production/engineering. Explain the *why* behind architectural choices. Use Spanish.
- **Validation Rigor**: Explicitly address **Data Leakage**. When working with medical or grouped data (e.g., CirCor), mandatorily use `StratifiedGroupKFold` to ensure patient-level splits. Generalization is the priority, not just metrics.

## 2. Notebook Structure & Styling
- **Minimal, High-Impact Markdown**: Keep Markdown cells extremely concise. Highlight crucial concepts, but avoid walls of text.
- **Pedagogical Narrative (Problem -> Solution)**: Before applying a solution (like Fine-Tuning), let students empirically see the *problem* (e.g., showing a random confusion matrix of an untrained head) so they understand the "why". Apply a "no-spoiler" approach: structure the class delivery to transition students from theoretical understanding to practical application, building curiosity for their final projects.
- **Granular Code Cells**: Separate code logically (data loading -> visualization -> model loading -> inference -> evaluation). Do not put everything into one massive cell.

## 3. Visualizations
- **Show, Don't Just Tell**: Always include visual proof.
- **Premium Aesthetics**: Use `plotly` (with `template='plotly_dark'` or similar premium themes) for interactive, high-impact plots like 3D PCA embeddings or Confusion Matrices. Use `matplotlib` primarily for pure image subplots to avoid overloading the browser.
- **Computer Vision**: Always plot original images vs. bounding boxes/masks. For real-time webcam inference (e.g., YOLO+SAM Live), strictly use `OpenCV` in standalone `.py` scripts to avoid freezing the Jupyter environment.
- **NLP/LLMs**:
  - Always clearly print the prompt going *into* the model and the raw text coming *out*, formatting it nicely.
  - **Tokenización**: Visualizar tokens con barras de colores usando `plotly` (un color por token, mostrar el texto fragmentado). Esto es más impactante que cualquier diapositiva estática.
  - **Matrices de atención**: Usar `BertViz` u herramientas similares para mostrar la atención interna de modelos reales. No reimplementar desde cero.
  - **Embeddings**: Visualizar con PCA/TSNE en 3D con `plotly`. Reusar el patrón ya establecido en `transfer_learning/`.

## 4. Code Quality
- **Reproducibility**: Set seeds (`pl.seed_everything(42)` or `random.seed`).
- **Comments**: Keep in-code comments concise but descriptive, especially noting tensor shapes or token limits.

## 5. Parcial 3 Specifics (Hardware & Resource Constraints)
- **Dataloaders & Windows**: When creating PyTorch `DataLoader`s, default to `num_workers=0` to prevent crashes/freezes on students' Windows machines. Use `pin_memory=True` and `persistent_workers=True` (when applicable) to optimize data loading.
- **Local LLMs**: Use **Ollama** as the primary LLM backend — it handles CUDA/Metal/CPU automatically, with no compilation needed. Preferred models: `gemma4:e4b` (~9.6 GB, for demo machine) and `gemma4:e2b` (~7.2 GB, for student laptops). Both are natively multimodal (texto + visión) — no separate projector files needed.
- **RAG Limits**: Keep context retrieval extremely small (Top 1 or 2 chunks maximum) to prevent overloading local context windows.
- **Embeddings**: Use `google-genai` with `gemini-embedding-2-preview` (free tier, 3072 dims, API key via `.env`). For offline fallback, `SentenceTransformers` (`all-MiniLM-L6-v2`) is a valid alternative — note it to students as a free, local option in `SETUP.md`.
- **Heavy Vision Models**: Prioritize lighter versions like `YOLO11n` (nano) o `FastSAM` / `MobileSAM` to ensure real-time inference viability on student machines.
  - **Estrategia YOLO+SAM**: Enseñar el uso de **Zero-shot Segmentation** mediante el paso de *Bounding Boxes* de YOLO como *prompts* de entrada para SAM. Es la forma más eficiente y pedagógica de conectar detección y segmentación.
- **Transfer Learning Strategy**:
  - **Paso 1: Linear Probe**: Congelar el backbone y entrenar solo la cabeza (MLP) para validar la calidad de las representaciones pre-entrenadas.
  - **Paso 2: Fine-Tuning**: Descongelar capas superiores solo si el Linear Probe ha convergido y el dataset es suficientemente grande para evitar overfitting.

## 6. Modularización y Estructura (Parcial 3)
- **Archivos Separados**: Para mantener las libretas limpias ("sin montañas de código"), la lógica pesada (limpieza, ingeniería de datos, PyTorch Lightning Modules, funciones de Plotly) **debe** aislarse en scripts de ayuda especializados. No sobrecargar un solo `helpers.py`.
- **El rol de la Libreta**: El Jupyter Notebook actúa como punto de consumo/explicación. Importa los helpers y construye el paso a paso ("Show, Don't Tell"). El nombre del notebook **NO** debe llevar prefijo numérico, usa el formato: `<Dataset> - <Tema>.ipynb` o `<Herramienta> - <Tema>.ipynb`.
- **Directorios**: Todo el Parcial 3 vive en `parcial_3/`. Las subcarpetas se nombran por el *Tema* (ej. `transfer_learning/`, `yolo/`, `rag/`), sin prefijos de semana para prever desfasamientos. No hay que cruzar dependencias entre temas.
- **Carpeta `local/`**: Cada submódulo puede tener una carpeta `local/` para pesos, datasets y assets pesados — está en `.gitignore`. Los notebooks deben incluir celdas de descarga automática si `local/` no existe.
- **Carpeta `helpers/`** (patrón RAG): Para módulos con más lógica reutilizable, crear una subcarpeta `helpers/` con scripts especializados (ej. `embeddings.py`, `retrieval.py`, `viz.py`). Esto enseña estructura profesional DRY.
- **Modelos CNN Recomendados**: Usar `resnet50` o `resnet101` (balance), `efficientnet_v2_s` (convergencia rápida), o `mobilenet_v3_large` (móvil). **Crítico:** usar siempre `weights='DEFAULT'` (no `pretrained=True`) para obtener la mejor versión (`IMAGENET1K_V2`).

## 7. RAG Module Specifics (`parcial_3/rag/`)
- **Secrets**: La API key de Gemini va en un archivo `.env` en la raíz del proyecto (ya está en `.gitignore`). Cargar con `python-dotenv`. **Nunca** hardcodear API keys en los notebooks.
- **Estructura de helpers**:
  - `helpers/embeddings.py` — Funciones para generar embeddings con Gemini (`gemini-embedding-2-preview`).
  - `helpers/retrieval.py` — Chunking de documentos y búsqueda vectorial con `numpy` (dot product, Top-K).
  - `helpers/viz.py` — Visualizaciones de tokens (barras Plotly), matrices de atención, y embeddings 3D.
- **Formato de documentos**: Los documentos para RAG van en `local/documents/` (`.txt` o `.pdf`). El chunking debe ser explícito y visible en el notebook (mostrar los chunks antes de embedear).
- **Prompt Engineering**: Siempre mostrar el template del prompt completo que se le envía al LLM — incluyendo el contexto recuperado. "Sin magia negra": el alumno debe ver exactamente qué texto entra al modelo.
- **Robustez en Adquisición**: Para RAG basado en APIs externas (ej. MET Museum), los scripts de ayuda **deben** implementar manejo de errores robusto: reintentos con `backoff`, gestión de `rate-limits` y validación de esquemas JSON para evitar fallos en la canalización de datos.
- **Scripts standalone**: El script `RAG - Chat Consola.py` sigue el mismo patrón que los scripts de YOLO (`Webcam - YOLO Live.py`): configurable al inicio con constantes en mayúsculas, bloque `if __name__ == '__main__'`.
- **Modelos Gemma 4 vía Ollama**:
  - Profesor/demo: `ollama pull gemma4:e4b` (~9.6 GB) — para RTX 3070
  - Estudiantes: `ollama pull gemma4:e2b` (~7.2 GB) — para laptops y Colab
  - Ambos son nativamente multimodales; Ollama gestiona CUDA/Metal sin configuración manual.
  - Para inferencia en notebooks: `import ollama; ollama.chat(model='gemma4:e4b', messages=[...])`