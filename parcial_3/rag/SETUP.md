# Setup Guide: Módulo RAG — `parcial_3/rag/`

## Índice

- [Prerrequisitos del Sistema](#prerrequisitos-del-sistema)
- [Paso 1: Activar el entorno virtual](#paso-1-activar-el-entorno-virtual)
- [Paso 2: Instalar dependencias](#paso-2-instalar-dependencias)
- [Paso 3: Instalar Ollama (Motor LLM)](#paso-3-instalar-ollama-motor-llm)
- [Paso 4: Descargar y Probar Gemma 4 (LLM Local)](#paso-4-descargar-y-probar-gemma-4-llm-local)
- [Paso 5: Configurar Gemini API (Embeddings)](#paso-5-configurar-gemini-api-embeddings)
- [Paso 6: Autenticarse en Hugging Face (Tokenizer)](#paso-6-autenticarse-en-hugging-face-tokenizer)

---

Antes de ejecutar cualquier notebook de este módulo, sigue estos pasos en orden para asegurar que tu entorno esté listo.

---

## Prerrequisitos del Sistema

> [!IMPORTANT]
> **Versión de Python:** Se recomienda usar **Python 3.12** o **3.13**.
> Se asume que ya clonaste el repositorio y creaste el entorno virtual `.venv` en la raíz (como se indica en el README principal).

---

## Paso 1: Activar el entorno virtual

Es fundamental trabajar dentro del entorno virtual para que las librerías no choquen con otras clases.

1. **Activar el entorno (Windows/PowerShell):**
```powershell
.\.venv\Scripts\activate
```
> **Resultado esperado:** Aparecerá un `(.venv)` verde al inicio de la línea en tu terminal, indicando que el entorno está activo.

---

## Paso 2: Instalar dependencias

Instalaremos las librerías necesarias para procesamiento de texto, modelos de Google y herramientas locales.

1. **Instalar librerías del módulo:**
```powershell
pip install "huggingface_hub>=1.0" transformers accelerate google-genai python-dotenv python-telegram-bot sentence-transformers ollama pypdf scikit-learn bertviz
```
> **Resultado esperado:** Verás barras de progreso instalando varias librerías. Al finalizar, mostrará un mensaje como `Successfully installed...` sin errores en rojo.

---

## Paso 3: Instalar Ollama (Motor LLM)

Ollama nos permite correr modelos de lenguaje potentes en nuestra propia computadora.

1. **Descarga e Instalación:**
   - Ve a **[ollama.com/download](https://ollama.com/download)**.
   - Descarga el instalador para tu sistema (Windows o Mac).
   - Ejecuta el archivo e instálalo.

2. **Comprobación rápida:**
Abre una terminal **nueva** y corre:
```powershell
ollama --version
```
> **Resultado esperado:** Deberías ver la versión instalada, por ejemplo: `ollama v0.1.x`.

---

## Paso 4: Descargar y Probar Gemma 4 (LLM Local)

Vamos a bajar el modelo **Gemma 4**, que es nativamente multimodal (entiende imágenes y texto).

1. **Descargar el modelo:**
(Si tu computadora es estándar o Mac Air, usa `e2b`. Si tienes GPU potente RTX, usa `e4b`)
```powershell
ollama pull gemma4:e2b
```
> **Resultado esperado:** Verás barras de progreso indicando `pulling manifest`, `downloading...` y finalmente `success`.

2. **Comprobación con Python:**
Crea un archivo temporal `test_ollama.py` con el siguiente código para verificar la conexión:

```python
import ollama

MODEL = "gemma4:e2b"

response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "¿Qué es un RAG en una frase simple?"}]
)

print(response.message.content)
```
> **Resultado esperado:** El script imprimirá una respuesta breve del modelo sobre qué es RAG.

---

## Paso 5: Configurar Gemini API (Embeddings)

Usaremos los **Embeddings** de Google Gemini para buscar información en documentos.

1. **Obtener API Key:**
   - Ve a **[Google AI Studio](https://aistudio.google.com/)**.
   - Haz clic en **"Get API Key"** y crea una.

2. **Configurar variable de entorno:**
En la raíz del repositorio, crea un archivo llamado `.env` y pega tu clave:
```text
GOOGLE_API_KEY=tu_clave_aqui
```

3. **Comprobación de conexión:**
Ejecuta este código para probar la API:

```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents="Hola mundo"
)

print(len(result.embeddings[0].values))
```
> **Resultado esperado:** El terminal imprimirá el número `3072`, que es la dimensión de los vectores de Gemini.

---

## Paso 6: Autenticarse en Hugging Face (Tokenizer)

Necesitamos el **Tokenizer** oficial para contar palabras/tokens con precisión.

1. **Aceptar términos (Web):**
   - Ve a: **[huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)**.
   - Haz clic en **"Accept"** o **"Request Access"**.

2. **Login en la terminal:**
```powershell
hf auth login
```
> **Resultado esperado:** La terminal te pedirá un "Token". Debes generarlo en tu cuenta de Hugging Face (Settings -> Tokens) con permiso "Read", pegarlo y dar Enter.

3. **Comprobación final:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokens = tokenizer("Hola, ¿cómo estás?")

print(len(tokens['input_ids']))
```
> **Resultado esperado:** Imprimirá un número (la cantidad de tokens), confirmando que tienes acceso al modelo.

---

¡Listo! Ya tienes todo el ecosistema configurado para empezar con RAG.
