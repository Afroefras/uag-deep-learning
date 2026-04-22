"""
documents.py — Helpers para extracción y chunking de documentos.

Funciones:
    - extract_pdf_text: Extrae texto crudo de un PDF con pypdf.
    - clean_text: Limpieza básica de texto extraído.
    - chunk_by_sections: Divide texto estructurado en chunks por headers markdown.
    - chunk_by_size: Divide texto en chunks de tamaño fijo con solapamiento.
    - print_chunks_preview: Imprime un preview de los chunks generados.
    - structure_pdf_with_llm: Pipeline completo PDF → texto estructurado via LLM.
"""

import re

try:
    from pypdf import PdfReader
except ImportError as e:
    raise ImportError(
        "pypdf no está instalado. Instálalo con: pip install pypdf"
    ) from e
from pathlib import Path


def extract_pdf_text(pdf_path: str | Path) -> dict:
    """
    Extrae el texto de cada página de un PDF usando pypdf.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Dict con claves:
            - "pages": lista de strings, uno por página.
            - "full_text": texto concatenado de todas las páginas.
            - "num_pages": número total de páginas.

    Ejemplo:
        from helpers.documents import extract_pdf_text
        result = extract_pdf_text("local/assets/sample_documents/Uno.pdf")
        print(result["num_pages"])
        print(result["pages"][0][:300])
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return {
        "pages": pages,
        "full_text": "\n\n".join(pages),
        "num_pages": len(pages),
    }


def clean_text(text: str) -> str:
    """
    Limpieza básica de texto extraído de PDF.

    Elimina:
        - Líneas con solo números (números de página)
        - Espacios en blanco excesivos
        - Caracteres de control

    Args:
        text: Texto crudo extraído del PDF.

    Returns:
        Texto limpio como string.
    """
    # Eliminar líneas que son solo números (páginas) o muy cortas
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.fullmatch(r"\d+", stripped) and len(stripped) > 2:
            clean_lines.append(stripped)

    # Unir y normalizar espacios
    cleaned = " ".join(clean_lines)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def chunk_by_sections(
    structured_text: str,
    header_pattern: str = r"^##\s+",
) -> list[dict]:
    """
    Divide un texto estructurado con headers markdown (## Sección) en chunks.

    Cada chunk representa una sección del documento.

    Args:
        structured_text: Texto con headers tipo "## Nombre de Sección".
        header_pattern: Regex que identifica las líneas de encabezado.

    Returns:
        Lista de dicts con claves:
            - "section": nombre de la sección (el header sin ##).
            - "content": contenido de la sección.
            - "char_count": número de caracteres.
            - "word_count": número de palabras aproximado.

    Ejemplo:
        from helpers.documents import chunk_by_sections
        chunks = chunk_by_sections(structured_text)
        for c in chunks:
            print(c["section"], "→", c["word_count"], "palabras")
    """
    lines = structured_text.split("\n")
    chunks = []
    current_section = "Introducción"
    current_lines = []

    for line in lines:
        if re.match(header_pattern, line, re.MULTILINE):
            # Guardar sección anterior si tiene contenido
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append({
                        "section": current_section,
                        "content": content,
                        "char_count": len(content),
                        "word_count": len(content.split()),
                    })
            # Iniciar nueva sección
            current_section = re.sub(r"^##\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Última sección
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({
                "section": current_section,
                "content": content,
                "char_count": len(content),
                "word_count": len(content.split()),
            })

    return chunks


def chunk_by_size(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict]:
    """
    Divide un texto en chunks de tamaño fijo medido en palabras, con solapamiento.

    Útil cuando el texto no tiene estructura de secciones clara.

    Args:
        text: Texto a dividir.
        chunk_size: Tamaño máximo de cada chunk en palabras.
        overlap: Número de palabras de solapamiento entre chunks consecutivos.

    Returns:
        Lista de dicts con claves "content", "char_count", "word_count", "chunk_index".

    Ejemplo:
        from helpers.documents import chunk_by_size
        chunks = chunk_by_size(long_text, chunk_size=300, overlap=50)
    """
    words = text.split()
    chunks = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        content = " ".join(words[start:end])
        chunks.append({
            "section": f"Chunk {idx + 1}",
            "content": content,
            "char_count": len(content),
            "word_count": len(words[start:end]),
            "chunk_index": idx,
        })
        start += chunk_size - overlap
        idx += 1

    return chunks


def print_chunks_preview(chunks: list[dict], n_chars: int = 200) -> None:
    """
    Imprime un preview legible de los chunks generados.

    Args:
        chunks: Lista de chunks generados por chunk_by_sections o chunk_by_size.
        n_chars: Número máximo de caracteres a mostrar del contenido.
    """
    print(f"Total de chunks: {len(chunks)}")
    print(f"{'─' * 65}")
    for i, chunk in enumerate(chunks):
        preview = chunk["content"][:n_chars].replace("\n", " ")
        if len(chunk["content"]) > n_chars:
            preview += "..."
        print(f"[{i:>2}] {chunk['section']}")
        print(f"     {chunk['word_count']} palabras | {chunk['char_count']} chars")
        print(f"     {preview!r}")
        print(f"{'─' * 65}")


def structure_pdf_with_llm(
    pdf_path: str | Path,
    output_path: str | Path,
    model: str = "gemma4:e4b",
    system_prompt: str | None = None,
    block_size: int = 10000,
    force: bool = False,
) -> str:
    """
    Pipeline completo: PDF → texto crudo → limpieza → LLM estructurador → texto con headers markdown.

    Si output_path ya existe y force=False, carga directamente del cache en disco.

    Args:
        pdf_path: Ruta al archivo PDF.
        output_path: Ruta donde guardar/cargar el texto estructurado (.txt).
        model: Modelo Ollama a usar para la estructuración.
        system_prompt: Prompt de sistema para el LLM estructurador.
            Si None, usa uno genérico orientado a documentos técnicos/académicos.
        block_size: Tamaño máximo de cada bloque de texto (en caracteres)
            que se envía al LLM. Documentos largos se dividen en bloques.
        force: Si True, re-procesa aunque ya exista el archivo de salida.

    Returns:
        El texto estructurado como string (con headers markdown ## Sección).

    Ejemplo:
        from helpers.documents import structure_pdf_with_llm
        texto = structure_pdf_with_llm(
            "local/assets/sample_documents/SAM33.pdf",
            "local/assets/sample_documents/SAM33_es.txt",
        )
    """
    try:
        import ollama
    except ImportError as e:
        raise ImportError(
            "ollama no está instalado. Instálalo con: pip install ollama"
        ) from e

    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    # ── Cache: si ya existe, cargar directamente ──
    if output_path.exists() and not force:
        texto = output_path.read_text(encoding="utf-8")
        print(f"✅ Cargando texto estructurado existente: {output_path}")
        print(f"   {len(texto):,} caracteres")
        return texto

    # ── Paso 1: Extracción cruda ──
    raw = extract_pdf_text(pdf_path)
    print(f"PDF: {pdf_path.name}  ({raw['num_pages']} páginas, {len(raw['full_text']):,} chars)")

    # ── Paso 2: Limpieza ──
    texto_limpio = clean_text(raw["full_text"])

    # ── Paso 3: Dividir en bloques ──
    bloques = [texto_limpio[i:i + block_size] for i in range(0, len(texto_limpio), block_size)]
    print(f"Texto limpio: {len(texto_limpio):,} chars → {len(bloques)} bloques de ~{block_size} chars")

    # ── Paso 4: Prompt del estructurador ──
    if system_prompt is None:
        system_prompt = (
            "Eres un estructurador de documentos experto. "
            "Recibirás fragmentos de un documento técnico/académico extraídos de un PDF. "
            "Tu tarea es: "
            "1. Reorganizar el texto en secciones claras con headers markdown (## Nombre de Sección). "
            "2. Traducir todo al español si está en inglés. "
            "3. Preservar toda la información importante (números, resultados, ecuaciones, referencias). "
            "4. Eliminar redundancias y texto de formato (números de página, headers repetidos). "
            "Responde SOLO con el texto estructurado, sin comentarios adicionales."
        )

    # ── Paso 5: Procesar con LLM ──
    print(f"Procesando con {model} — esto puede tomar unos minutos...")
    secciones = []
    for i, bloque in enumerate(bloques):
        print(f"  Bloque {i + 1}/{len(bloques)}...", end="\r")
        r = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Procesa este fragmento:\n\n{bloque}"},
            ],
            options={"temperature": 0.3, "top_p": 0.95, "top_k": 64},
        )
        secciones.append(r.message.content)

    texto_estructurado = "\n\n".join(secciones)

    # ── Paso 6: Guardar en disco ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(texto_estructurado, encoding="utf-8")
    print(f"\n✅ Texto estructurado guardado: {output_path}")
    print(f"   {len(texto_estructurado):,} caracteres")

    return texto_estructurado
