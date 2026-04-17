"""
viz.py — Helpers de visualización para el módulo RAG.

Funciones:
    - plot_tokens_colored: Visualización de tokens como barras de colores (Plotly).
    - plot_embeddings_3d: Visualización PCA/TSNE en 3D de embeddings (Plotly).
    - plot_chunk_stats: Gráfica de barras con estadísticas de chunks (Plotly).
    - plot_attention_heatmap: Mapa de calor de pesos de atención (Plotly).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    from sklearn.decomposition import PCA
except ImportError as e:
    raise ImportError(
        "scikit-learn no está instalado. Instálalo con: pip install scikit-learn"
    ) from e


# ── Paleta de colores para tokens ──────────────────────────────────────────────
_TOKEN_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#aecbfa", "#fdcfe8", "#b7e1a1", "#ffd966", "#d4a1e5",
    "#a8d8ea", "#f9b4ab", "#fae3d9", "#bbdfc8", "#f0e6ef",
]


def plot_tokens_colored(tokens: list[str], title: str = "Visualización de Tokens") -> go.Figure:
    """
    Muestra cada token como una barra de color diferente con el texto encima.

    Args:
        tokens: Lista de strings, cada uno es un token (ej. ["▁Hola", ",", "▁¿cómo"])
        title: Título del gráfico.

    Returns:
        Figura de Plotly lista para mostrar con fig.show().

    Ejemplo:
        from helpers.viz import plot_tokens_colored
        fig = plot_tokens_colored(["▁Hola", ",", "▁mundo"])
        fig.show()
    """
    n = len(tokens)
    colors = [_TOKEN_COLORS[i % len(_TOKEN_COLORS)] for i in range(n)]

    # Reemplazar caracteres especiales para visualización limpia
    display_tokens = [t.replace("▁", "·") for t in tokens]

    fig = go.Figure()

    for i, (tok, col) in enumerate(zip(display_tokens, colors)):
        fig.add_trace(go.Bar(
            x=[tok],
            y=[1],
            marker_color=col,
            text=tok,
            textposition="inside",
            insidetextanchor="middle",
            name=f"[{i}] {tok}",
            hovertemplate=f"<b>Token #{i}</b><br>Texto: '{tok}'<br>Índice: {i}<extra></extra>",
            width=0.85,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(
            title="Tokens",
            tickfont=dict(size=11),
            showgrid=False,
        ),
        yaxis=dict(visible=False),
        bargap=0.1,
        height=280,
        margin=dict(l=30, r=30, t=60, b=40),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
    )
    return fig


def plot_token_comparison(
    texts: list[str],
    tokenizers: dict,  # {"Nombre": tokenizer_object}
    title: str = "Comparación de Tokenizaciones",
) -> go.Figure:
    """
    Compara cuántos tokens genera cada tokenizer para una lista de textos.

    Args:
        texts: Lista de textos a tokenizar.
        tokenizers: Dict {"Nombre": tokenizer_object}.
        title: Título del gráfico.

    Returns:
        Figura de Plotly tipo barra agrupada.

    Ejemplo:
        from helpers.viz import plot_token_comparison
        fig = plot_token_comparison(
            ["Hola mundo", "Hello world", "def suma(a, b): return a + b"],
            {"Gemma 2": tokenizer_gemma, "GPT-2": tokenizer_gpt2},
        )
        fig.show()
    """
    fig = go.Figure()

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for idx, (name, tok) in enumerate(tokenizers.items()):
        counts = [len(tok(t)["input_ids"]) for t in texts]
        fig.add_trace(go.Bar(
            name=name,
            x=texts,
            y=counts,
            marker_color=colors[idx % len(colors)],
            text=counts,
            textposition="outside",
            hovertemplate=f"<b>{name}</b><br>Texto: %{{x}}<br>Tokens: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        template="plotly_dark",
        barmode="group",
        xaxis=dict(title="Texto", tickfont=dict(size=10)),
        yaxis=dict(title="Número de tokens"),
        height=420,
        margin=dict(l=40, r=40, t=70, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
    )
    return fig


def plot_embeddings_3d(
    embeddings: np.ndarray,
    labels: list[str],
    colors: list[str] | None = None,
    title: str = "Embeddings en 3D (PCA)",
) -> go.Figure:
    """
    Proyecta embeddings de alta dimensión a 3D usando PCA y los muestra en Plotly.

    Args:
        embeddings: Array (N, D) con los embeddings.
        labels: Lista de N strings con el nombre/texto de cada punto.
        colors: Lista de N strings de colores (hexadecimal o nombre). Si None, usa la paleta interna.
        title: Título del gráfico.

    Returns:
        Figura de Plotly 3D lista para mostrar.

    Ejemplo:
        from helpers.viz import plot_embeddings_3d
        fig = plot_embeddings_3d(emb_array, ["Texto 1", "Texto 2", ...])
        fig.show()
    """
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(embeddings)

    if colors is None:
        colors = [_TOKEN_COLORS[i % len(_TOKEN_COLORS)] for i in range(len(labels))]

    variance = pca.explained_variance_ratio_
    subtitle = (
        f"PC1={variance[0]:.1%} | PC2={variance[1]:.1%} | PC3={variance[2]:.1%} "
        f"| Total={sum(variance):.1%} varianza explicada"
    )

    fig = go.Figure(data=go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(
            size=8,
            color=colors,
            opacity=0.85,
            line=dict(width=0.5, color="white"),
        ),
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", font=dict(size=16, color="white")),
        template="plotly_dark",
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
            bgcolor="#1e1e2e",
        ),
        height=600,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="#1e1e2e",
    )
    return fig


def plot_chunk_stats(
    chunks: list[dict],
    title: str = "Estadísticas de Chunks",
) -> go.Figure:
    """
    Gráfica de barras horizontal con word_count y char_count por sección.

    Args:
        chunks: Lista de dicts con "section", "word_count" y "char_count".
        title: Título del gráfico.

    Returns:
        Figura de Plotly lista para mostrar con fig.show().

    Ejemplo:
        from helpers.viz import plot_chunk_stats
        fig = plot_chunk_stats(chunks)
        fig.show()
    """
    section_names = [c["section"][:35] for c in chunks]
    word_counts = [c["word_count"] for c in chunks]
    char_counts = [c["char_count"] for c in chunks]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=section_names,
        x=word_counts,
        name="Palabras",
        orientation="h",
        marker_color="#4e79a7",
        text=word_counts,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Palabras: %{x}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=section_names,
        x=char_counts,
        name="Caracteres",
        orientation="h",
        marker_color="#f28e2b",
        text=char_counts,
        textposition="outside",
        visible="legendonly",
        hovertemplate="<b>%{y}</b><br>Caracteres: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        template="plotly_dark",
        xaxis=dict(title="Conteo"),
        yaxis=dict(autorange="reversed"),
        height=max(300, 40 * len(chunks)),
        margin=dict(l=200, r=60, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
    )
    return fig


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: list[str],
    title: str = "Mapa de Atención",
    layer: int = 0,
    head: int = 0,
) -> go.Figure:
    """
    Visualiza una matriz de atención (N_tokens x N_tokens) como mapa de calor.

    Args:
        attention_matrix: Array (N, N) con los pesos de atención (ya softmaxxeados).
        tokens: Lista de N strings con los tokens correspondientes.
        title: Título del gráfico.
        layer: Número de capa (solo para el título).
        head: Número de cabeza de atención (solo para el título).

    Returns:
        Figura de Plotly de mapa de calor.
    Example:
        from helpers.viz import plot_attention_heatmap
        fig = plot_attention_heatmap(attn[0][layer][head].numpy(), tokens)
        fig.show()
    """
    # Limpiamos caracteres raros de visualización
    display_tokens = [t.replace("▁", "·") for t in tokens]

    # Creamos el mapa de calor
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=display_tokens,
        y=display_tokens,
        # CAMBIO 1: Paleta de colores más vibrante y contrastada para fondo oscuro
        colorscale="Blues",
        hovertemplate="<b>Query: %{y}</b><br>Key: %{x}<br>Atención: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Peso de atención"),
    ))

    # Actualizamos el diseño
    fig.update_layout(
        title=dict(
            text=f"{title} — Capa {layer}, Cabeza {head}",
            font=dict(size=16),
        ),
        xaxis=dict(title="Key (a qué mira)", tickangle=-45),
        yaxis=dict(title="Query (quién mira)", autorange="reversed"),
        height=500,
        margin=dict(l=80, r=40, t=80, b=100),
    )
    return fig


def plot_transformer_attention(
    model,
    tokenizer,
    text: str,
    layer: int = None,
    head: int = None,
    find_connection: list[str] = None,
    title: str = "Atención en el Transformer",
) -> go.Figure:
    """
    Realiza la inferencia y visualiza la atención de un modelo tipo Transformer.
    Si se provee 'find_connection', busca automáticamente la capa/cabeza con mayor atención.

    Args:
        model: Modelo de Hugging Face (ej. bert_model).
        tokenizer: Tokenizador (ej. bert_tokenizer).
        text: La frase a analizar.
        layer: Capa específica a visualizar (0-indexed).
        head: Cabeza específica a visualizar (0-indexed).
        find_connection: Opcional. Lista de 2 strings [query, key] para buscar la mejor conexión.
        title: Título del gráfico.

    Returns:
        tuple: (figura_plotly, atenciones_tupla, lista_tokens)
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch no está instalado. Instálalo con: pip install torch"
        ) from e

    # 1. Tokenización e Inferencia
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise ValueError(
            "El modelo no devolvió atenciones. Asegúrate de cargarlo con 'output_attentions=True'."
        )

    atenciones = outputs.attentions  # Tupla de tensores (1, heads, seq, seq)

    # 2. Búsqueda automática de la mejor conexión
    mejor_capa, mejor_cabeza = 0, 0
    if find_connection and len(find_connection) == 2:
        try:
            # Buscamos los índices de los tokens
            idx_q = tokens.index(find_connection[0])
            idx_k = tokens.index(find_connection[1])

            mejor_peso = -1
            for l in range(len(atenciones)):
                for h in range(atenciones[l].shape[1]):
                    # Peso: Capa L, Cabeza H, Token Q mira a Token K
                    peso = atenciones[l][0, h, idx_q, idx_k].item()
                    if peso > mejor_peso:
                        mejor_peso = peso
                        mejor_capa, mejor_cabeza = l, h

            # Si no se especificó capa/cabeza, usamos la mejor encontrada
            if layer is None: layer = mejor_capa
            if head is None: head = mejor_cabeza

            print(f"🕵️‍♂️ ¡Magia encontrada! Para '{find_connection[0]}' -> '{find_connection[1]}':")
            print(f"   Usa la Capa {mejor_capa}, Cabeza {mejor_cabeza} (Atención: {mejor_peso*100:.1f}%)")

        except ValueError:
            print(f"⚠️ No se encontró la pareja {find_connection} en los tokens: {tokens}")
            if layer is None: layer = 0
            if head is None: head = 0
    else:
        # Defaults si no se especifica nada
        if layer is None: layer = 0
        if head is None: head = 0

    # 3. Visualización usando la función existente
    attn_matrix = atenciones[layer][0, head].cpu().numpy()

    fig = plot_attention_heatmap(
        attn_matrix,
        tokens,
        title=title,
        layer=layer,
        head=head,
    )

    return fig, atenciones, tokens
