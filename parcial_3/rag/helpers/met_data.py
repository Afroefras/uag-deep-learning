import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display, HTML


# API del Metropolitan Museum of Art (pública, sin API key)
MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

def fetch_met_query(query: str, n_target: int = 10, dept_id: int = 11) -> list[dict]:
    """
    Busca pinturas en el MET por término temático.
    dept_id=11 = European Paintings (Renacimiento → s.XX).
    """
    r = requests.get(MET_BASE + "/search", params={
        "hasImages": "true",
        "isPublicDomain": "true",
        "departmentId": dept_id,
        "q": query,
    }, timeout=15)
    ids = r.json().get("objectIDs") or []

    results = []
    for obj_id in ids:
        if len(results) >= n_target:
            break
        try:
            obj = requests.get(MET_BASE + f"/objects/{obj_id}", timeout=8).json()
            img = obj.get("primaryImageSmall", "")
            if not img or not obj.get("isPublicDomain"):
                continue
            tags = [t["term"] for t in (obj.get("tags") or [])]
            results.append({
                "artist":    obj.get("artistDisplayName", "Unknown"),
                "title":     obj.get("title", "Untitled"),
                "style":     query.title(),
                "year":      obj.get("objectDate", ""),
                "tags":      ", ".join(tags[:4]),
                "image_url": img,
            })
        except Exception:
            continue
    return results


def check_url(url: str, timeout: int = 8) -> bool:
    """Devuelve True si la URL responde con código 200."""
    try:
        r = requests.head(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def validate_urls_parallel(urls: list[str], max_workers: int = 12) -> list[bool]:
    """
    Valida todas las URLs en paralelo usando threads.
    Mucho más rápido que hacerlo de forma secuencial.
    """
    results = [False] * len(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_url, url): i for i, url in enumerate(urls)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results


def show_gallery(df_sample, n=9, fig_width=240):
    """Muestra una galería de imágenes desde las URLs del DataFrame."""
    sample = df_sample.sample(n=min(n, len(df_sample)), random_state=42)
    cells = []
    for _, row in sample.iterrows():
        style_color = "#7c5cbf" if "Abstract" in row["style"] else (
            "#2e7d32" if "Impress" in row["style"] else (
            "#c62828" if "Baroque" in row["style"] else "#1565c0"))
        cell = f"""
        <div style='display:inline-block; margin:8px; text-align:center; vertical-align:top; width:{fig_width}px'>
            <img src='{row['image_url']}' width='{fig_width}' height='180'
                 style='object-fit:cover; border-radius:8px; border:2px solid {style_color}'>
            <div style='font-size:11px; color:#ccc; margin-top:5px'>
                <b style='color:#fff'>{row['artist'].split(',')[0]}</b><br>
                <i>{row['title'][:35]}{'...' if len(row['title'])>35 else ''}</i><br>
                <span style='color:#aaa'>{row['year']}</span>&nbsp;
                <span style='color:{style_color}; font-size:10px'>● {row['style']}</span>
            </div>
        </div>"""
        cells.append(cell)
    display(HTML(f"<div style='background:#1e1e2e; padding:12px; border-radius:12px'>{''.join(cells)}</div>"))


def look_paintings_by_property(df: pd.DataFrame, questions: list[dict],col: str, umbral: float = 0.0, top_n: int = 6):
    pregunta_info = next((p for p in questions if p["label"] == col), None)
    if pregunta_info is None:
        print(f"Opciones: {[p['label'] for p in questions]}")
        return

    res = (
        df[df[col] >= umbral]
        .sort_values(col, ascending=False)
        .head(top_n)
    )
    print(f"🔍  {pregunta_info['display']}")
    print(f"    Umbral: {umbral:.2f}  →  {len(res)} resultado(s)")

    cells = []
    for _, row in res.iterrows():
        sc  = row[col]
        pct = int(sc * 100)
        hue = int(sc * 120)
        bc  = f"hsl({hue},80%,45%)"
        cell = f"""
        <div style='display:inline-block;margin:10px;text-align:center;vertical-align:top;width:250px'>
            <img src='{row['image_url']}' width='250' height='170'
                 style='object-fit:cover;border-radius:8px;border:2.5px solid {bc}'>
            <div style='font-size:11px;color:#ccc;margin-top:5px'>
                <b style='color:#fff'>{row['artist'].split(',')[0]}</b><br>
                <i>{row['title'][:35]}{'...' if len(row['title'])>35 else ''}</i><br>
                <span style='color:#aaa'>{row['style']}</span>
                <div style='margin:5px auto;background:#333;border-radius:4px;height:9px;width:200px'>
                    <div style='background:{bc};width:{min(pct*2,200)}px;height:9px;border-radius:4px'></div>
                </div>
                <span style='color:{bc};font-weight:bold'>{sc:.3f}</span>
            </div>
        </div>"""
        cells.append(cell)
    display(HTML(f"<div style='background:#1e1e2e;padding:12px;border-radius:12px'>{''.join(cells)}</div>"))

