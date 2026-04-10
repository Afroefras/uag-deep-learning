# Setup Guide: Módulo RAG — `parcial_3/rag/`

Antes de ejecutar cualquier notebook de este módulo, sigue estos pasos en orden. Puedes hacerlo tú mismo (preferible para aprender) — solo necesitas seguir los comandos de terminal.

---

## Prerequisitos del Sistema

- Python instalado (3.10, 3.11, o 3.12)
- `venv` ya creado en la raíz del repositorio

---

## Paso 1: Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

---

## Paso 2: Instalar dependencias del módulo RAG

```powershell
pip install "huggingface_hub>=1.0" transformers accelerate google-genai python-dotenv python-telegram-bot sentence-transformers ollama pypdf scikit-learn bertviz
```

> **Nota**: Es importante instalar `huggingface_hub>=1.0` explícitamente. Versiones anteriores (como la `0.1.2`) no incluyen el CLI de autenticación ni muchos comandos modernos.

> **Nota sobre `sentence-transformers`**: Es una alternativa completamente gratuita y offline a Gemini Embeddings. No requiere API key — solo internet para descargar el modelo la primera vez (~90MB). Ideal si no tienes internet o quieres evitar el API key. Su uso es: `from sentence_transformers import SentenceTransformer; model = SentenceTransformer("all-MiniLM-L6-v2")`.

---

## Paso 2.5: Autenticarse en Hugging Face (para Gemma 2 Tokenizer)

Este paso es necesario si quieres usar el Tokenizer oficial de Google directamente con `transformers`. No es necesario para Ollama ni para Gemini Embeddings.

### A. Aceptar los términos de Gemma en la web
1. Abre tu navegador e inicia sesión en huggingface.co con tu cuenta.
2. Ve directamente a: **[huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)**
3. Verás un formulario de acceso ("Gated model"). Completa los datos y haz clic en **"Accept"**.
4. La aprobación es prácticamente instantánea.

> ⚠️ El login desde la terminal **no es suficiente**. Primero hay que aceptar los términos en la web. Solo se hace una vez por cuenta.

### B. Autenticarse desde la terminal

El CLI anterior (`huggingface-cli`) está deprecado. El nuevo es `hf`:

```powershell
hf auth login
```

Te pedirá un token. Créalo en: **Hugging Face → Settings → Access Tokens → New token (Read)**.

**Verificar que el login fue exitoso:**
```powershell
hf auth whoami
```

---

## Paso 3: Instalar Ollama (motor de LLMs locales)

Ollama gestiona CUDA/Metal automáticamente, sin compilación ni dependencias de Visual Studio.

### 🪟 Windows
1. Descarga el instalador desde **[ollama.com/download](https://ollama.com/download)**
2. Ejecuta el `.exe` e instala (un clic)
3. Ollama queda corriendo como servicio en segundo plano automáticamente

### 🍎 Mac (Apple Silicon y Intel)
1. Descarga desde **[ollama.com/download](https://ollama.com/download)** → opción Mac
2. Arrastra `Ollama.app` a tu carpeta de Aplicaciones y ábrela

> **Verificar instalación:** Abre una terminal nueva y corre:
> ```powershell
> ollama --version
> ```

---

## Paso 4: Descargar los modelos Gemma 4


> [!IMPORTANT]
> **Todos los modelos de Gemma 4 son nativamente multimodales** (texto + imagen). No se necesita archivo `mmproj` separado — Ollama lo maneja internamente. Es una de las ventajas clave frente a `llama-cpp-python`.

### Para el Profesor (RTX 3070, ~8GB VRAM)

```powershell
# Gemma 4 E4B — 9.6 GB, 128K contexto, Text + Image
ollama pull gemma4:e4b
```

> ⚠️ **Nota de VRAM**: Con 8GB de VRAM puede ir justo. Si tienes problemas, usa `gemma4:e2b` que funciona igual pedagógicamente y pesa 7.2 GB.

### Para Estudiantes (laptops con GPU o Colab)

```bash
# Gemma 4 E2B — 7.2 GB, 128K contexto, Text + Image
ollama pull gemma4:e2b
```

### Verificar que los modelos están descargados

```powershell
ollama list
```

Deberías ver algo como:
```
NAME            ID              SIZE    MODIFIED
gemma4:e4b      ...             9.6 GB  ...
```

---

## Paso 5: Configurar API Key de Gemini (para Embeddings)

1. Ve a [Google AI Studio](https://aistudio.google.com/) e inicia sesión con tu cuenta Google.
2. Haz clic en **"Get API Key"** → **"Create API key"**.
3. Copia la clave generada.
4. En la raíz del repositorio, crea un archivo `.env`:

```
GOOGLE_API_KEY=tu_clave_aqui
```

> El archivo `.env` ya está en `.gitignore` — nunca se subirá al repositorio. **Nunca compartas esta clave.**

**Verificar que funciona:**
```python
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Hola, estoy probando los embeddings de Gemini."
)
print(f"✅ Embeddings OK — Dimensiones: {len(result.embeddings[0].values)}")
# Esperado: ✅ Embeddings OK — Dimensiones: 3072
```

---

## Paso 6: Verificaciones finales

Ejecuta estas verificaciones en orden antes de empezar con los notebooks:

**Verificar Ollama texto:**
```python
import ollama

response = ollama.chat(
    model="gemma4:e4b",  # o gemma4:e2b
    messages=[{"role": "user", "content": "¿Cuántos planetas tiene el sistema solar? Responde en una línea."}]
)
print(response.message.content)
```

**Verificar Ollama multimodal (visión):**
```python
import ollama

response = ollama.chat(
    model="gemma4:e4b",
    messages=[{
        "role": "user",
        "content": "Describe esta imagen en español en 2 oraciones.",
        "images": ["ruta/a/tu/imagen.jpg"]  # path local o URL
    }]
)
print(response.message.content)
```

**Verificar transformers (requiere Paso 2.5 completado):**
```python
from transformers import AutoTokenizer

# Requiere: (1) haber aceptado los términos en huggingface.co/google/gemma-2-2b
#           (2) haber corrido 'hf auth login' en la terminal
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokens = tokenizer("Hola, ¿cómo estás?")
print(f"✅ Transformers OK — {len(tokens['input_ids'])} tokens")
# Esperado: ✅ Transformers OK — 9 tokens
```

> Si aún no has completado el Paso 2.5, puedes verificar `transformers` con un modelo abierto sin restricciones:
> ```python
> from transformers import AutoTokenizer
> tokenizer = AutoTokenizer.from_pretrained("gpt2")  # sin login
> print(f"✅ Transformers OK — {len(tokenizer('Hola mundo')['input_ids'])} tokens")
> ```
