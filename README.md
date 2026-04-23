# UAG Deep Learning: De la Teoría a Producción (Parcial 3)

Este repositorio contiene el material y las libretas Jupyter para la clase de **Deep Learning** de últimos semestres (Actuaría / Ciencia de Datos) en la Universidad Autónoma de Guadalajara (UAG). 

El objetivo principal de esta etapa (Parcial 3) es la transición de conceptos puramente académicos hacia entornos y despliegues orientados a la producción y la ingeniería de IA.

## 🚀 Temario y Enfoque

- **Entrenamiento Optimizado**: Uso estricto de **PyTorch Lightning** (`pl.LightningModule`) para abstraer ciclos de entrenamiento, crear código limpio y escalable (e.g., Transfer Learning con arquitecturas CNN modernas).
- **Inferencia y Despliegue (SOTA)**: Consumo de APIs de IA Generativa (como **Google Gemini**) y modelos locales de Visión por Computadora (**YOLO11**, **SAM**) y NLP (**Gemma**, **Llama** vía `llama-cpp-python`).
- **Arquitectura de Transformers y LLMs**: Exploración práctica de *tokenizers*, visualización del mecanismo de atención y manejo de memoria conversacional (stateful vs stateless).
- **RAG y Búsqueda Semántica Multimodal**: Construcción de *pipelines* de Retrieval-Augmented Generation (RAG) combinando documentos locales (PDFs) y consumo de APIs (como el Museo MET), utilizando *embeddings* multimodales.
- **Narrativa "Show, Don't Tell"**: Énfasis en la visualización interactiva de resultados tangibles utilizando `plotly` y en crear bases vectoriales ligeras que corran eficientemente de manera local.

## 🛠 Entorno de Desarrollo

> [!IMPORTANT]
> **Versión de Python recomendada:** Para evitar problemas al instalar librerías científicas (como `numpy` o `contourpy`), te recomendamos fuertemente usar **Python 3.12** o **Python 3.13**. Si tienes una versión más reciente (como Python 3.14) y experimentas errores de instalación, la mejor opción es desinstalar esa versión e instalar Python 3.12 o 3.13 desde la página oficial de Python.

Se **DEBE** usar un entorno virtual para manejar las dependencias del proyecto de forma aislada (Regla de la clase). Sigue estos pasos en orden:

1. **Clonar el repositorio** (solo la primera vez):
```powershell
git clone https://github.com/Afroefras/uag-deep-learning.git
```
> **Resultado esperado:** Verás texto indicando que se está descargando, como `Cloning into 'uag-deep-learning'...` y el progreso de descarga.

2. **Entrar a la carpeta del proyecto**:
```powershell
cd uag-deep-learning
```
> **Resultado esperado:** La ruta de tu terminal cambiará y ahora terminará en `\uag-deep-learning>`.

3. **Crear el entorno virtual** (dentro de la carpeta del proyecto):
```powershell
python -m venv .venv
```
> **Resultado esperado:** *Tardará unos segundos y no mostrará ningún mensaje al terminar*, pero si revisas tus archivos verás que se creó una nueva carpeta llamada `.venv`.

4. **Activar el entorno**:
```powershell
.\.venv\Scripts\activate
```
> **Resultado esperado:** Aparecerá un `(.venv)` verde al inicio de la línea en tu terminal, indicando que el entorno está activo.

5. **Instalar los requerimientos**:
```powershell
pip install -r requirements.txt
```
> **Resultado esperado:** Verás texto corriendo y barras de progreso instalando varias librerías. Al finalizar, mostrará un mensaje como `Successfully installed numpy... pandas...` sin errores en rojo.
## 🌳 Flujo de Trabajo (Git Workflow)

El repositorio está diseñado para evolucionar progresivamente a lo largo del periodo.

- **Rama Principal (Default)**: `2026-01`
- **Ramas por Tema**: El profesor creará una rama específica para preparar cada tema nuevo. Una vez lista (y antes de la clase), se fusionará ("merge") en la rama `2026-01`.

### 👨‍🎓 Instrucciones para Alumnos
Para estar siempre al día con la clase tienes dos opciones:

**Opción A (Recomendada si no modificas mucho el repo)**:
Simplemente sitúate en la rama principal y actualiza antes de cada clase:

1. **Asegúrate de estar en la rama principal**:
```powershell
git checkout 2026-01
```
> **Resultado esperado:** Si ya estabas ahí, dirá `Already on '2026-01'`. Si estabas en otra rama, dirá `Switched to branch '2026-01'`.

2. **Descarga los cambios más recientes**:
```powershell
git pull origin 2026-01
```
> **Resultado esperado:** Si había temas nuevos, verás una lista de los archivos actualizados. Si ya estabas al día, dirá `Already up to date`.

**Opción B (Recomendada si haces tus propios experimentos y notas)**:
Crea tu propia rama de estudio personal y actualízala mezclando los cambios del profesor conforme sube nuevos temas.

1. **Al inicio del semestre, crea tu propia rama**:
```powershell
git checkout -b alumno/mi-nombre
```
> **Resultado esperado:** Verás un mensaje que dice `Switched to a new branch 'alumno/mi-nombre'`.

2. **Cuando el profesor avise de un tema nuevo, descarga los cambios a tu rama**:
```powershell
git pull origin 2026-01
```
> **Resultado esperado:** Verás cómo se integran los archivos nuevos del profesor a tu rama personal. Te mostrará un resumen de los archivos que fueron añadidos o modificados.

> [!CAUTION]
> **Política de Datos:** NO subas ("commit") archivos pesados (pesos `.pt`, bases de datos, copias masivas de imágenes directas en carpetas root) a GitHub. Usa las funciones de descarga incluidas y descarga localmente lo que necesiten tus libretas. **ASEGURATE DE AGREGAR EN EL .gitignore LOS ARCHIVOS QUE NO DEBEN SER SUBIDOS**
