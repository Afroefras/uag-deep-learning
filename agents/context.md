# Context for AI Agents: PyTorch Lightning Deep Learning Notebooks

This repository contains Jupyter Notebooks used for teaching Deep Learning to last-semester undergraduate students. When creating new notebooks or modifying existing ones, future AI agents **must** strictly adhere to the following pedagogical and styling guidelines:

## 1. Content and Framework
- **Core Framework**: All models should be implemented using **PyTorch Lightning** (`pytorch_lightning as pl`). Avoid raw PyTorch training loops (use `pl.LightningModule` and `pl.Trainer`).
- **Target Audience**: Last-semester Deep Learning students. The content should be advanced but pedagogically structured. Explain the *why* behind architectural choices rather than the absolute basics of programming. 
- **Language**: Use Spanish, as it is the language of instruction for the professor.

## 2. Notebook Structure & Styling
- **Minimal, High-Impact Markdown**: Keep Markdown cells extremely concise. Do not include too much markdown text unless it is highly relevant (e.g. highlighting a crucial concept, clarifying an architectural decision, or separating major sections). Avoid walls of text.
- **Granular Code Cells**: Separate code cells logically to show the step-by-step process of implementing the solution (e.g., data loading, data visualization, model definition, training, evaluation). Do not put everything into one massive cell. This helps students digest the material progressively.

## 3. Visualizations
- **Show, Don't Just Tell**: Always include visual proof of the data, the process, and the model's performance. Focus heavily on visualizations.
- **Data Exploration**: Print shapes and visualize a batch of the data before building the model (e.g., using `matplotlib.pyplot` to show images, audio waves, text samples, etc.).
- **Evaluation**: Visualize predictions (e.g., side-by-side true vs. predicted labels) and always include robust evaluation metrics. Use tools like `sklearn.metrics.confusion_matrix` and `classification_report` to show results.

## 4. Code Quality
- **Self-Contained Executions**: Always start with necessary installation commands (e.g., `%pip install lightning` if needed for Colab compatibility) and imports.
- **Reproducibility**: Set seeds using `pl.seed_everything(42)` (or similar) at the beginning of the notebook so students can replicate results.
- **Comments**: Keep in-code comments concise but descriptive. They should explain the logic flow (e.g. the transformations or dimensions of tensors passing through PyTorch layers).
