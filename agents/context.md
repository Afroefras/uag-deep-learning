# Context for AI Agents: Deep Learning & Production Notebooks (Parcial 3)

This repository contains Jupyter Notebooks used for teaching Deep Learning to last-semester undergraduate students (actuaries/data scientists). When creating new notebooks or modifying existing ones, future AI agents **must** strictly adhere to the following pedagogical and styling guidelines:

## 1. Content and Frameworks (Training vs. Inference)
- **Training**: When building or fine-tuning models from scratch (e.g., CNNs, ResNet, EfficientNet, simple Transformers), strictly use **PyTorch Lightning** (`pytorch_lightning as pl`). Avoid raw PyTorch training loops.
- **Inference & State-of-the-Art**: When teaching inference with modern tools (YOLO, SAM, Hugging Face, LLMs), **DO NOT** wrap them in PyTorch Lightning. Use their native APIs and pipelines (e.g., `ultralytics` for YOLO, `transformers` pipelines for HF, `llama-cpp-python` for local LLMs).
- **Target Audience**: Last-semester students with strong mathematical backgrounds but transitioning into production/engineering. Explain the *why* behind architectural choices. Use Spanish.

## 2. Notebook Structure & Styling
- **Minimal, High-Impact Markdown**: Keep Markdown cells extremely concise. Highlight crucial concepts, but avoid walls of text. 
- **Granular Code Cells**: Separate code logically (data loading -> visualization -> model loading -> inference -> evaluation). Do not put everything into one massive cell.

## 3. Visualizations
- **Show, Don't Just Tell**: Always include visual proof. 
- **Computer Vision**: Always plot original images vs. bounding boxes/masks using `plotly`.
- **NLP/LLMs**: Clearly print the prompt going *into* the model and the raw text coming *out*, formatting it nicely.

## 4. Code Quality
- **Reproducibility**: Set seeds (`pl.seed_everything(42)` or `random.seed`).
- **Comments**: Keep in-code comments concise but descriptive, especially noting tensor shapes or token limits.

## 5. Parcial 3 Specifics (Hardware & Resource Constraints)
- **Local LLMs**: Always default to lightweight, quantized models (e.g., GGUF format, Q4_K_M quantization) suitable for CPU/limited RAM.
- **RAG Limits**: Keep context retrieval extremely small (Top 1 or 2 chunks maximum) to prevent overloading local context windows. Use simple tools (`numpy` dot product, `SentenceTransformers`) over heavy databases unless otherwise specified.
- **Heavy Vision Models**: Prioritize lighter versions like `YOLOv8n` (nano) or `FastSAM` / `MobileSAM` to ensure real-time inference viability on student machines.