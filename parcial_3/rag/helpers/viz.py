"""
viz.py — Helpers de visualización para el módulo RAG.

Funciones:
    - plot_tokens_colored: Visualización de tokens como barras de colores (Plotly).
    - plot_embeddings_3d: Visualización PCA/TSNE en 3D de embeddings (Plotly).
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

    Ejemplo:
        from helpers.viz import plot_attention_heatmap
        fig = plot_attention_heatmap(attn[0][layer][head].numpy(), tokens)
        fig.show()
    """
    display_tokens = [t.replace("▁", "·") for t in tokens]

    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=display_tokens,
        y=display_tokens,
        colorscale="Viridis",
        hovertemplate="<b>Query: %{y}</b><br>Key: %{x}<br>Atención: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Peso de atención"),
    ))

    fig.update_layout(
        title=dict(
            text=f"{title} — Capa {layer}, Cabeza {head}",
            font=dict(size=16, color="white"),
        ),
        template="plotly_dark",
        xaxis=dict(title="Key (a qué mira)", tickangle=-45),
        yaxis=dict(title="Query (quién mira)", autorange="reversed"),
        height=500,
        margin=dict(l=80, r=40, t=80, b=100),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
    )
    return fig
