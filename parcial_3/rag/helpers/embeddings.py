"""
embeddings.py — Helpers para generar embeddings con Gemini.

Funciones:
    - get_client: Inicializa el cliente de Gemini con la API key del .env.
    - embed_text: Genera un embedding para un texto individual.
    - embed_batch: Genera embeddings para una lista de textos.
    - save_embeddings: Guarda embeddings en disco (numpy .npy).
    - load_embeddings: Carga embeddings guardados.
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

try:
    from google import genai
except ImportError as e:
    raise ImportError(
        "google-genai no está instalado. Instálalo con: pip install google-genai"
    ) from e

# Modelo de embeddings
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMS  = 3072


def get_client():
    """
    Inicializa y devuelve el cliente de Gemini cargando la API key desde .env.

    Returns:
        genai.Client listo para usar.

    Raises:
        ValueError: Si GOOGLE_API_KEY no está definida en el .env.

    Ejemplo:
        from helpers.embeddings import get_client
        client = get_client()
    """
    # Buscar el .env en la raíz del repo (2 niveles arriba de helpers/)
    env_path = Path().cwd().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fallback: buscar en el CWD

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY no encontrada. Crea un archivo .env en la raíz del repo "
            "con la línea: GOOGLE_API_KEY=tu_clave_aqui"
        )
    return genai.Client(api_key=api_key)


def embed_text(text: str, client=None) -> np.ndarray:
    """
    Genera un embedding para un texto individual usando gemini-embedding-001.

    Args:
        text: El texto a embedear.
        client: Cliente de Gemini (si None, lo crea automáticamente).

    Returns:
        Array numpy de forma (3072,) con los valores del embedding.

    Ejemplo:
        from helpers.embeddings import embed_text
        vec = embed_text("El UNO es un juego de cartas")
        print(vec.shape)  # (3072,)
    """
    if client is None:
        client = get_client()

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)


def embed_batch(
    texts: list[str],
    client=None,
    verbose: bool = True,
    delay: float = 0.5,
) -> np.ndarray:
    """
    Genera embeddings para una lista de textos.

    Intenta primero con una sola petición al API (batch nativo).
    Si falla por límites, cae back a modo secuencial con pausa entre peticiones.

    Args:
        texts: Lista de strings a embeddear.
        client: Cliente de Gemini. Si None, lo crea.
        verbose: Si True, muestra progreso.
        delay: Segundos de pausa entre peticiones en modo secuencial (fallback).
            Aumenta este valor si sigues obteniendo errores 429.

    Returns:
        Array numpy de forma (N, 3072) donde N = len(texts).

    Ejemplo:
        from helpers.embeddings import embed_batch
        vecs = embed_batch(["texto 1", "texto 2", "texto 3"])
        print(vecs.shape)  # (3, 3072)
    """
    import time

    if client is None:
        client = get_client()

    # ── Intento 1: batch nativo (1 sola petición HTTP) ──
    try:
        if verbose:
            print(f"  Generando {len(texts)} embeddings en una sola petición...")
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        vecs = np.stack(
            [np.array(e.values, dtype=np.float32) for e in result.embeddings],
            axis=0,
        )
        if verbose:
            print(f"  ✅ {len(texts)} embeddings generados ({EMBEDDING_DIMS} dims c/u)")
        return vecs

    except Exception as e:
        if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
            raise  # error distinto, no silenciar
        if verbose:
            print(f"  ⚠️  Batch nativo falló (429). Cambiando a modo secuencial (delay={delay}s)...")

    # ── Fallback: secuencial con pausa ──
    embeddings = []
    for i, text in enumerate(texts):
        if verbose:
            print(f"  Embeddiendo chunk {i + 1}/{len(texts)}...", end="\r")
        vec = embed_text(text, client=client)
        embeddings.append(vec)
        if i < len(texts) - 1:
            time.sleep(delay)

    if verbose:
        print(f"  ✅ {len(texts)} embeddings generados ({EMBEDDING_DIMS} dims c/u)    ")

    return np.stack(embeddings, axis=0)


def save_embeddings(
    embeddings: np.ndarray,
    path: str | Path,
) -> None:
    """
    Guarda una matriz de embeddings en disco en formato numpy (.npy).

    Args:
        embeddings: Array numpy de forma (N, D).
        path: Ruta destino del archivo .npy.

    Ejemplo:
        from helpers.embeddings import save_embeddings
        save_embeddings(vecs, "local/embeddings/uno_chunks.npy")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embeddings)
    print(f"  Embeddings guardados en: {path}  ({embeddings.shape})")


def load_embeddings(path: str | Path) -> np.ndarray:
    """
    Carga embeddings guardados con save_embeddings.

    Args:
        path: Ruta al archivo .npy.

    Returns:
        Array numpy de forma (N, D).

    Ejemplo:
        from helpers.embeddings import load_embeddings
        vecs = load_embeddings("local/embeddings/uno_chunks.npy")
    """
    return np.load(str(path))
