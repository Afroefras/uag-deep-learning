"""
multimodal.py — Helpers para embeddings multimodales con gemini-embedding-2-preview.

Funciones:
    - embed_image_url:   Descarga una imagen por URL y genera su embedding.
    - embed_images_batch: Procesa N imágenes con rate-limiting y caché en .npy.
    - embed_text_query:  Genera el embedding de una pregunta/query de texto.
    - cosine_similarity_matrix: Calcula la matriz de similitudes query × imágenes.
    - show_matches:      Muestra las imágenes que superan un umbral para una query.
"""

import io
import time
import numpy as np
from pathlib import Path

try:
    import requests
except ImportError as e:
    raise ImportError(
        "requests no está instalado. Instálalo con: pip install requests"
    ) from e

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "google-genai no está instalado. Instálalo con: pip install google-genai"
    ) from e

try:
    from PIL import Image as PILImage
except ImportError as e:
    raise ImportError(
        "Pillow no está instalado. Instálalo con: pip install pillow"
    ) from e

from .embeddings import get_client, save_embeddings, load_embeddings

# Modelo multimodal (distinto al de texto)
MULTIMODAL_MODEL  = "gemini-embedding-2-preview"
MULTIMODAL_DIMS   = 3072

# Tiempo de espera entre peticiones para evitar 429
_DEFAULT_DELAY = 2.0  # segundos
_MAX_RETRIES   = 3


# ── Descarga de imágenes ────────────────────────────────────────────────────────

def _download_image_bytes(url: str, timeout: int = 10) -> tuple[bytes, str]:
    """
    Descarga una imagen desde una URL y devuelve (bytes, mime_type).
    Convierte a JPEG si no es JPEG/PNG para compatibilidad con la API.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Detectar tipo MIME
    ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    raw_bytes = resp.content

    # Normalizar a JPEG para la API (más compatible)
    img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


# ── Embedding de una imagen ─────────────────────────────────────────────────────

def embed_image_url(url: str, client=None) -> np.ndarray:
    """
    Descarga una imagen por URL y genera su embedding con gemini-embedding-2-preview.

    Args:
        url:    URL directa a la imagen.
        client: Cliente de Gemini (si None, lo crea automáticamente).

    Returns:
        Array numpy de forma (3072,).

    Ejemplo:
        from helpers.multimodal import embed_image_url
        vec = embed_image_url("https://uploads.wikiart.org/.../starry-night.jpg!Large.jpg")
        print(vec.shape)  # (3072,)
    """
    if client is None:
        client = get_client()

    img_bytes, mime_type = _download_image_bytes(url)

    result = client.models.embed_content(
        model=MULTIMODAL_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
        ],
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)


# ── Embedding de lote de imágenes (con caché) ───────────────────────────────────

def embed_images_batch(
    urls: list[str],
    cache_path: str | Path | None = None,
    client=None,
    verbose: bool = True,
    delay: float = _DEFAULT_DELAY,
) -> np.ndarray:
    """
    Genera embeddings para una lista de URLs de imágenes.

    Si cache_path existe, carga los embeddings desde disco (no llama a la API).
    Si no existe, genera los embeddings y los guarda en cache_path.

    Args:
        urls:       Lista de URLs de imágenes.
        cache_path: Ruta al archivo .npy de caché. Si None, no cachea.
        client:     Cliente de Gemini. Si None, lo crea.
        verbose:    Si True, muestra progreso.
        delay:      Segundos de pausa entre peticiones (evita 429).

    Returns:
        Array numpy de forma (N, 3072).

    Ejemplo:
        from helpers.multimodal import embed_images_batch
        vecs = embed_images_batch(urls, cache_path="local/embeddings/wikiart.npy")
        print(vecs.shape)  # (50, 3072)
    """
    # ── Checkpoint: cargar desde caché si existe ──
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            if verbose:
                print(f"✅ Embeddings cargados desde caché: {cache_path}")
            return load_embeddings(cache_path)

    if client is None:
        client = get_client()

    embeddings = []
    n = len(urls)

    for i, url in enumerate(urls):
        if verbose:
            print(f"  [{i+1}/{n}] Embeddeando imagen...", end="\r")

        # Reintentos con backoff exponencial
        for attempt in range(_MAX_RETRIES):
            try:
                vec = embed_image_url(url, client=client)
                embeddings.append(vec)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = delay * (2 ** attempt)
                    if verbose:
                        print(f"\n  ⚠️  429 en imagen {i+1}. Esperando {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    if verbose:
                        print(f"\n  ❌ Error en imagen {i+1} (intento {attempt+1}): {e}")
                    if attempt == _MAX_RETRIES - 1:
                        # Insertar vector cero para no romper la matriz
                        embeddings.append(np.zeros(MULTIMODAL_DIMS, dtype=np.float32))
        else:
            embeddings.append(np.zeros(MULTIMODAL_DIMS, dtype=np.float32))

        if i < n - 1:
            time.sleep(delay)

    result = np.stack(embeddings, axis=0)

    if verbose:
        print(f"  ✅ {n} embeddings generados ({MULTIMODAL_DIMS} dims c/u)    ")

    # Guardar en caché
    if cache_path is not None:
        save_embeddings(result, cache_path)

    return result


# ── Embedding de texto (query semántica) ────────────────────────────────────────

def embed_text_query(query: str, client=None) -> np.ndarray:
    """
    Genera el embedding de un texto (pregunta semántica) usando el mismo
    modelo multimodal, permitiendo comparación directa con embeddings de imagen.

    Args:
        query:  La pregunta o descripción semántica.
        client: Cliente de Gemini. Si None, lo crea.

    Returns:
        Array numpy de forma (3072,).

    Ejemplo:
        from helpers.multimodal import embed_text_query
        vec = embed_text_query("¿Parece un paisaje tranquilo?")
        print(vec.shape)  # (3072,)
    """
    if client is None:
        client = get_client()

    result = client.models.embed_content(
        model=MULTIMODAL_MODEL,
        contents=[query],
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)


def embed_text_queries(queries: list[str], client=None, verbose: bool = True) -> np.ndarray:
    """
    Genera embeddings para múltiples preguntas semánticas en una sola petición.

    Args:
        queries: Lista de strings (preguntas semánticas).
        client:  Cliente de Gemini. Si None, lo crea.
        verbose: Si True, imprime confirmación.

    Returns:
        Array numpy de forma (Q, 3072) donde Q = len(queries).
    """
    if client is None:
        client = get_client()

    if verbose:
        print(f"  Generando {len(queries)} embeddings de texto...")

    result = client.models.embed_content(
        model=MULTIMODAL_MODEL,
        contents=queries,
    )
    vecs = np.stack(
        [np.array(e.values, dtype=np.float32) for e in result.embeddings],
        axis=0,
    )
    if verbose:
        print(f"  ✅ {len(queries)} embeddings de queries listos")
    return vecs


# ── Similitud coseno ────────────────────────────────────────────────────────────

def cosine_similarity_matrix(
    query_vecs: np.ndarray,
    image_vecs: np.ndarray,
) -> np.ndarray:
    """
    Calcula la similitud coseno entre Q queries y N imágenes.

    Args:
        query_vecs: Array (Q, D) con los embeddings de las preguntas.
        image_vecs: Array (N, D) con los embeddings de las imágenes.

    Returns:
        Array (Q, N) con scores de similitud en [0, 1].

    Ejemplo:
        from helpers.multimodal import cosine_similarity_matrix
        scores = cosine_similarity_matrix(query_vecs, image_vecs)
        # scores[i, j] = similitud entre query i e imagen j
    """
    # Normalizar
    q_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
    i_norm = image_vecs / (np.linalg.norm(image_vecs, axis=1, keepdims=True) + 1e-8)

    # Producto punto = similitud coseno para vectores normalizados
    return q_norm @ i_norm.T  # (Q, N)
