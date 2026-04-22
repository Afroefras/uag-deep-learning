# Setup Guide: Módulo RAG — `parcial_3/rag/`

## Índice

- [Prerequisitos del Sistema](#prerequisitos-del-sistema)
- [Paso 1: Activar el entorno virtual](#paso-1-activar-el-entorno-virtual)
- [Paso 2: Instalar dependencias](#paso-2-instalar-dependencias)
- [Paso 3: Instalar Ollama (Motor LLM)](#paso-3-instalar-ollama-motor-llm)
- [Paso 4: Descargar y Probar Gemma 4 (LLM Local)](#paso-4-descargar-y-probar-gemma-4-llm-local)
- [Paso 5: Configurar Gemini API (Embeddings)](#paso-5-configurar-gemini-api-embeddings)
- [Paso 6: Autenticarse en Hugging Face (Tokenizer)](#paso-6-autenticarse-en-hugging-face-tokenizer)

---

Antes de ejecutar cualquier notebook de este módulo, sigue estos pasos en orden para asegurar que tu entorno esté listo.

---

## Prerequisitos del Sistema

- Python instalado (3.10, 3.11, o 3.12)
- `venv` ya creado en la raíz del repositorio

---

## Paso 1: Activar el entorno virtual

Es fundamental trabajar dentro del entorno virtual para que las librerías no choquen con otras clases.

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

---

## Paso 2: Instalar dependencias

Instalaremos todas las librerías necesarias para procesamiento de texto, modelos de Google y herramientas locales.

```powershell
pip install "huggingface_hub>=1.0" transformers accelerate google-genai python-dotenv python-telegram-bot sentence-transformers ollama pypdf scikit-learn bertviz
```

> **Nota**: Es importante instalar `huggingface_hub>=1.0` explícitamente para evitar problemas con comandos de autenticación modernos.

---

## Paso 3: Instalar Ollama (Motor LLM)

Ollama es la herramienta que nos permite correr modelos de lenguaje potentes en nuestra propia computadora.

### 🪟 Windows
1. Descarga el instalador desde **[ollama.com/download](https://ollama.com/download)**
2. Ejecuta el `.exe` e instala.
3. Ollama queda corriendo como servicio en el "tray" (junto al reloj) automáticamente.

### 🍎 Mac
1. Descarga desde **[ollama.com/download](https://ollama.com/download)** → opción Mac.
2. Arrastra `Ollama.app` a Aplicaciones y ábrela.

### ✅ Comprobación rápida
Abre una terminal **nueva** y corre:
```powershell
ollama --version
```
> Deberías ver la versión instalada (ej. `ollama v0.1.x`).

---

## Paso 4: Descargar y Probar Gemma 4 (LLM Local)

¡Esta es la parte más emocionante! Vamos a bajar el cerebro del modelo **Gemma 4**, que es nativamente multimodal (entiende imágenes y texto).

### A. Descargar el modelo
Elige el tamaño según tu computadora:

- **GPU potente (RTX 3060+, ~8GB VRAM):**
  ```powershell
  ollama pull gemma4:e4b
  ```
- **Computadora estándar (Laptop / Mac Air):**
  ```powershell
  ollama pull gemma4:e2b
  ```

### ✅ Comprobación con Python
Crea un archivo temporal `test_ollama.py` o corre esto en un notebook:

```python
import ollama

# Cambia a "gemma4:e2b" si bajaste el más chico
MODEL = "gemma4:e4b" 

print(f"🤖 Probando {MODEL}...")
response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "¿Qué es un RAG en una frase simple?"}]
)
print(f"Respuesta: {response.message.content}")
```

---

## Paso 5: Configurar Gemini API (Embeddings)

Para buscar información en documentos (RAG), usaremos los **Embeddings** de Google Gemini, que son muy precisos y rápidos.

1. Ve a **[Google AI Studio](https://aistudio.google.com/)**.
2. Haz clic en **"Get API Key"** → **"Create API key"**.
3. En la raíz del repositorio, crea un archivo llamado `.env` con este contenido:
   ```
   GOOGLE_API_KEY=tu_clave_aqui
   ```

### ✅ Comprobación con Python
```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Probar los embeddings
result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents="Hola mundo"
)
print(f"✅ Gemini Embeddings OK — Dimensiones: {len(result.embeddings[0].values)}")
# Deberías ver: Dimensiones: 3072
```

---

## Paso 6: Autenticarse en Hugging Face (Tokenizer)

Aunque corramos el modelo localmente, a veces necesitamos el **Tokenizer** oficial desde Hugging Face para contar palabras/tokens con precisión.

### A. Aceptar términos en la web (Obligatorio)
1. Ve a: **[huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)**.
2. Si ves un botón de **"Accept"** o **"Request Access"**, dale clic (es instantáneo).

### B. Login en la terminal
```powershell
hf auth login
```
Usa tu token de **Hugging Face → Settings → Tokens** (debe ser tipo "Read").

### ✅ Comprobación con Python
```python
from transformers import AutoTokenizer

# Esto fallará si no hiciste el login arriba o no aceptaste los términos en la web
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokens = tokenizer("Hola, ¿cómo estás?")
print(f"✅ Tokenizer OK — {len(tokens['input_ids'])} tokens generados.")
```

---

¡Listo! Ya tienes todo el ecosistema configurado para empezar con RAG.
