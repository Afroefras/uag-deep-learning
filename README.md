# UAG Deep Learning: De la Teoría a Producción (Parcial 3)

Este repositorio contiene el material y las libretas Jupyter para la clase de **Deep Learning** de últimos semestres (Actuaría / Ciencia de Datos) en la Universidad Autónoma de Guadalajara (UAG). 

El objetivo principal de esta etapa (Parcial 3) es la transición de conceptos puramente académicos hacia entornos y despliegues orientados a la producción y la ingeniería de IA.

## 🚀 Temario y Enfoque

- **Entrenamiento Optimizado**: Uso estricto de **PyTorch Lightning** (`pl.LightningModule`) para abstraer ciclos de entrenamiento, crear código limpio y escalable (e.g., Transfer Learning con arquitecturas CNN modernas).
- **Inferencia State-of-the-Art (SOTA)**: Consumo de APIs nativas y modelos avanzados de Visión por Computadora (**YOLOv8**, **SAM / FastSAM**) y NLP/LLMs (**Hugging Face**, **llama-cpp-python**).
- **Módulos Locales y Escalables**: Trabajo con modelos ligeros (GGUF, Q4_K_M) y bases vectoriales simples para RAG, garantizando que el diseño pueda correr eficientemente en las computadoras portátiles de los estudiantes y luego escale hacia la nube.
- **Narrativa "Show, Don't Tell"**: Énfasis en la visualización interactiva de resultados tangibles utilizando `plotly`.

## 🛠 Entorno de Desarrollo

Se **DEBE** usar un entorno virtual para manejar las dependencias del proyecto de forma aislada (Regla de la clase).

```powershell
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activar el entorno
.\.venv\Scripts\activate

# 3. Instalar los requerimientos
pip install -r requirements.txt
```

## 🌳 Flujo de Trabajo (Git Workflow)

El repositorio está diseñado para evolucionar progresivamente a lo largo del periodo.

- **Rama Principal (Default)**: `2026-01`
- **Ramas por Tema**: El profesor creará una rama específica para preparar cada tema nuevo. Una vez lista (y antes de la clase), se fusionará ("merge") en la rama `2026-01`.

### 👨‍🎓 Instrucciones para Alumnos
Para estar siempre al día con la clase tienes dos opciones:

**Opción A (Recomendada si no modificas mucho el repo)**:
Simplemente sitúate en la rama principal y actualiza antes de cada clase:
```powershell
git checkout 2026-01
git pull origin 2026-01
```

**Opción B (Recomendada si haces tus propios experimentos y notas)**:
Crea tu propia rama de estudio personal y actualízala mezclando los cambios del profesor conforme sube nuevos temas:
```powershell
# Al inicio del semestre, crea tu rama:
git checkout -b alumno/mi-nombre

# Cuando el profesor avise que subió un tema nuevo a la rama default:
git pull origin 2026-01
```

> [!CAUTION]
> **Política de Datos:** NO subas ("commit") archivos pesados (pesos `.pt`, bases de datos, copias masivas de imágenes directas en carpetas root) a GitHub. Usa las funciones de descarga incluidas y descarga localmente lo que necesiten tus libretas. **ASEGURATE DE AGREGAR EN EL .gitignore LOS ARCHIVOS QUE NO DEBEN SER SUBIDOS**
