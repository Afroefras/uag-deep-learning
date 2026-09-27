"""
retrieval.py — Helpers para búsqueda vectorial y construcción del prompt RAG.

Funciones:
    - cosine_similarity: Similitud coseno entre un vector query y una matriz.
    - retrieve_top_k: Recupera los K chunks más relevantes para un query.
    - build_rag_prompt: Construye el prompt completo (contexto + pregunta) para el LLM.
    - format_retrieval_results: Imprime los resultados de búsqueda de forma legible.
    - rag_query: Pipeline RAG completo (query → embed → retrieve → generate).
    - compare_with_without_rag: Compara respuesta SIN RAG vs CON RAG.
"""

import numpy as np


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud coseno entre un vector de consulta y cada fila de una matriz.

    La similitud coseno mide el ángulo entre vectores (no la magnitud):
        sim(A, B) = (A · B) / (|A| * |B|)

    Args:
        query_vec: Vector de forma (D,) — el embedding del query.
        matrix: Matriz de forma (N, D) — los embeddings del corpus.

    Returns:
        Array de forma (N,) con las similitudes (entre -1 y 1, típicamente 0 a 1).

    Ejemplo:
        from helpers.retrieval import cosine_similarity
        sims = cosine_similarity(query_embed, corpus_embeds)
        print(sims.shape)  # (num_chunks,)
    """
    # Normalizar query
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    # Normalizar cada fila del corpus
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    corpus_norm = matrix / norms
    # Producto punto → similitudes
    return corpus_norm @ q_norm


def retrieve_top_k(
    query_vec: np.ndarray,
    corpus_embeddings: np.ndarray,
    chunks: list[dict],
    k: int = 2,
) -> list[dict]:
    """
    Recupera los K chunks más similares al query usando similitud coseno.

    Args:
        query_vec: Embedding del query, forma (D,).
        corpus_embeddings: Embeddings del corpus, forma (N, D).
        chunks: Lista de dicts con "section" y "content" (mismo orden que corpus_embeddings).
        k: Número de chunks a recuperar.

    Returns:
        Lista de hasta K dicts, cada uno con:
            - "rank": posición en el ranking (1 = más relevante).
            - "score": similitud coseno (float, 0 a 1).
            - "section": nombre de la sección.
            - "content": contenido del chunk.

    Ejemplo:
        from helpers.retrieval import retrieve_top_k
        resultados = retrieve_top_k(query_embed, corpus_embeds, chunks, k=2)
        for r in resultados:
            print(r["rank"], r["score"]:.3f, r["section"])
    """
    sims = cosine_similarity(query_vec, corpus_embeddings)
    # Índices de mayor a menor similitud
    top_indices = np.argsort(sims)[::-1][:k]

    resultados = []
    for rank, idx in enumerate(top_indices, 1):
        resultados.append({
            "rank": rank,
            "score": float(sims[idx]),
            "section": chunks[idx]["section"],
            "content": chunks[idx]["content"],
        })
    return resultados


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict],
    system_instruction: str | None = None,
) -> str:
    """
    Construye el prompt completo que se enviará al LLM incluyendo el contexto recuperado.

    Sigue el patrón estándar de RAG:
        [Instrucción del sistema]
        --- CONTEXTO RECUPERADO ---
        [Chunk 1]
        [Chunk 2]
        --------------------------
        PREGUNTA: [query]

    Args:
        query: La pregunta del usuario.
        retrieved_chunks: Lista de chunks devuelta por retrieve_top_k.
        system_instruction: Instrucción de sistema personalizada. Si None, usa una por defecto.

    Returns:
        String con el prompt completo listo para enviar al LLM.

    Ejemplo:
        from helpers.retrieval import build_rag_prompt
        prompt = build_rag_prompt("¿Cuántas cartas se reparten?", resultados)
        print(prompt)
    """
    if system_instruction is None:
        system_instruction = (
            "Eres un asistente experto. "
            "Responde ÚNICAMENTE basándote en el contexto proporcionado. "
            "Si la información no está en el contexto, di 'No encontré esa información en el documento.' "
            "Responde siempre en español, de forma clara y concisa."
        )

    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Fuente: {chunk['section']} — relevancia: {chunk['score']:.2%}]\n"
            f"{chunk['content']}"
        )
    context_str = "\n\n".join(context_parts)

    prompt = (
        f"{system_instruction}\n\n"
        f"{'─' * 50}\n"
        f"CONTEXTO RECUPERADO:\n\n"
        f"{context_str}\n\n"
        f"{'─' * 50}\n"
        f"PREGUNTA: {query}"
    )
    return prompt


def format_retrieval_results(results: list[dict], query: str) -> None:
    """
    Imprime los resultados de búsqueda de forma legible para el notebook.

    Args:
        results: Lista de chunks devuelta por retrieve_top_k.
        query: El query original (para contexto).
    """
    print(f"Query  : {query!r}")
    print(f"{'─' * 65}")
    for r in results:
        bar_len = int(r["score"] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"Rank #{r['rank']} | Score: {r['score']:.4f} | {bar}")
        print(f"        Sección: {r['section']}")
        preview = r["content"][:180].replace("\n", " ")
        if len(r["content"]) > 180:
            preview += "..."
        print(f"        {preview!r}")
        print(f"{'─' * 65}")


def rag_query(
    query: str,
    corpus_embeddings,
    chunks: list[dict],
    model: str = "gemma4:e4b",
    system_instruction: str | None = None,
    k: int = 2,
    client=None,
    verbose: bool = True,
) -> dict:
    """
    Pipeline RAG completo: query → embed → retrieve top-K → generate con LLM.

    Args:
        query: La pregunta del usuario.
        corpus_embeddings: Array numpy (N, D) con los embeddings del corpus.
        chunks: Lista de dicts con "section" y "content".
        model: Modelo Ollama a usar para la generación.
        system_instruction: Instrucción de sistema para el prompt RAG.
        k: Número de chunks a recuperar.
        client: Cliente de Gemini para embeddings. Si None, lo crea.
        verbose: Si True, imprime información del proceso.

    Returns:
        Dict con claves:
            - "query": la pregunta original.
            - "answer": la respuesta generada por el LLM.
            - "retrieved": los chunks recuperados (lista de dicts).
            - "prompt": el prompt completo enviado al LLM.

    Ejemplo:
        from helpers.retrieval import rag_query
        result = rag_query("¿Cuántas clases tiene el modelo?", embeddings, chunks)
        print(result["answer"])
    """
    try:
        import ollama
    except ImportError as e:
        raise ImportError(
            "ollama no está instalado. Instálalo con: pip install ollama"
        ) from e

    from helpers.embeddings import embed_text, get_client

    if client is None:
        client = get_client()

    # 1. Embed query
    query_embed = embed_text(query, client=client)

    # 2. Retrieve top-K
    retrieved = retrieve_top_k(query_embed, corpus_embeddings, chunks, k=k)

    if verbose:
        format_retrieval_results(retrieved, query)

    # 3. Build prompt
    prompt = build_rag_prompt(query, retrieved, system_instruction=system_instruction)

    # 4. Generate
    r = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "top_p": 0.95, "top_k": 64},
    )

    return {
        "query": query,
        "answer": r.message.content,
        "retrieved": retrieved,
        "prompt": prompt,
    }


def compare_with_without_rag(
    query: str,
    corpus_embeddings,
    chunks: list[dict],
    model: str = "gemma4:e4b",
    system_instruction: str | None = None,
    k: int = 2,
    client=None,
) -> dict:
    """
    Compara la respuesta de un LLM SIN contexto vs CON contexto RAG.

    Útil para demostrar el valor de RAG: el mismo LLM, la misma pregunta,
    pero con y sin los chunks recuperados.

    Args:
        query: La pregunta del usuario.
        corpus_embeddings: Array numpy (N, D) con los embeddings del corpus.
        chunks: Lista de dicts con "section" y "content".
        model: Modelo Ollama a usar para la generación.
        system_instruction: Instrucción de sistema para el prompt RAG.
        k: Número de chunks a recuperar.
        client: Cliente de Gemini para embeddings. Si None, lo crea.

    Returns:
        Dict con claves:
            - "query": la pregunta original.
            - "sin_rag": respuesta sin contexto.
            - "con_rag": respuesta con contexto (dict completo de rag_query).

    Ejemplo:
        from helpers.retrieval import compare_with_without_rag
        cmp = compare_with_without_rag("¿Qué arquitectura usa?", embeddings, chunks)
        print("SIN RAG:", cmp["sin_rag"])
        print("CON RAG:", cmp["con_rag"]["answer"])
    """
    try:
        import ollama
    except ImportError as e:
        raise ImportError(
            "ollama no está instalado. Instálalo con: pip install ollama"
        ) from e

    from helpers.embeddings import get_client

    if client is None:
        client = get_client()

    # SIN RAG
    r_sin = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Responde en español: {query}",
        }],
        options={"temperature": 0.3, "top_p": 0.95, "top_k": 64},
    )

    # CON RAG
    con_rag = rag_query(
        query, corpus_embeddings, chunks,
        model=model, system_instruction=system_instruction,
        k=k, client=client, verbose=False,
    )

    return {
        "query": query,
        "sin_rag": r_sin.message.content,
        "con_rag": con_rag,
    }
