"""
helpers/video_utils.py
======================
Utilidades para el manejo de archivos de video.

Dependencias externas: NINGUNA.
Solo usa módulos de la biblioteca estándar de Python (subprocess, os, pathlib).

Requisito del sistema:
    - `ffmpeg` debe estar instalado y accesible desde la terminal.
    - En Windows: instalar desde https://ffmpeg.org/download.html
      y agregarlo al PATH del sistema.
    - Verificar instalación corriendo en terminal: ffmpeg -version
"""

import subprocess
import os
from pathlib import Path


def extract_audio_from_video(
    video_path: str,
    output_wav_path: str = None,
    sample_rate: int = 22050,
    channels: int = 1,
) -> str:
    """
    Extrae el audio de un archivo de video (MP4, MKV, AVI, etc.)
    y lo guarda como un archivo WAV usando ffmpeg vía subprocess.

    No se requiere ninguna librería de Python adicional; solo ffmpeg
    instalado en el sistema operativo.

    Args:
        video_path (str):
            Ruta al archivo de video de entrada (ej. "mi_clase.mp4").

        output_wav_path (str, optional):
            Ruta donde se guardará el archivo .wav resultante.
            Si no se especifica, se usa el mismo directorio y nombre
            del video de entrada (ej. "mi_clase.wav").

        sample_rate (int, optional):
            Frecuencia de muestreo del audio de salida en Hz.
            22050 es suficiente para análisis de audio/ML.
            Usa 44100 si requieres calidad de CD completa.
            Por defecto: 22050.

        channels (int, optional):
            Número de canales de audio.
            1 = Mono (recomendado para entrenar CNNs, reduce dimensionalidad).
            2 = Estéreo.
            Por defecto: 1 (Mono).

    Returns:
        str: La ruta absoluta del archivo .wav generado.

    Raises:
        FileNotFoundError: Si el archivo de video no existe.
        RuntimeError: Si ffmpeg no está instalado o el proceso falla.

    Ejemplo de uso:
        >>> ruta_wav = extract_audio_from_video("clase_01.mp4")
        >>> print(ruta_wav)
        /ruta/absoluta/clase_01.wav

        >>> # Especificando todos los parámetros:
        >>> ruta_wav = extract_audio_from_video(
        ...     video_path="videos/clase_02.mp4",
        ...     output_wav_path="audios/clase_02.wav",
        ...     sample_rate=44100,
        ...     channels=2
        ... )
    """
    # --- Validación de entrada ---
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de video: {video_path}")

    # Si no se especifica ruta de salida, la generamos automáticamente
    # cambiando la extensión del archivo de entrada a .wav
    if output_wav_path is None:
        output_wav_path = video_path.with_suffix(".wav")
    else:
        output_wav_path = Path(output_wav_path).resolve()

    # Creamos el directorio de destino si no existe
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Construcción del comando ffmpeg ---
    # Explicación de cada bandera:
    #   -i <video>     : archivo de entrada (input)
    #   -vn            : sin video en la salida (video none)
    #   -acodec pcm_s16le : códec de audio PCM 16-bit Little Endian (WAV estándar)
    #   -ar <rate>     : sample rate de salida
    #   -ac <channels> : número de canales de audio
    #   -y             : sobreescribir el archivo de salida si ya existe
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-y",
        str(output_wav_path),
    ]

    print(f"⚙️  Extrayendo audio de: {video_path.name}")
    print(f"   Guardando en        : {output_wav_path}")

    # --- Ejecución del proceso ---
    # capture_output=True evita que ffmpeg imprima en la terminal del notebook.
    # text=True decodifica stdout/stderr como texto legible.
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    # Verificamos que ffmpeg terminó sin errores (código de retorno 0)
    if result.returncode != 0:
        # Si ffmpeg no está instalado, el error dice "not found" / "no reconocido"
        if "not found" in result.stderr.lower() or "no se reconoce" in result.stderr.lower():
            raise RuntimeError(
                "ffmpeg no está instalado o no está en el PATH del sistema.\n"
                "Descárgalo desde: https://ffmpeg.org/download.html"
            )
        raise RuntimeError(
            f"ffmpeg terminó con un error (código {result.returncode}):\n{result.stderr}"
        )

    print(f"✅ Audio extraído exitosamente ({sample_rate} Hz, {'Mono' if channels == 1 else 'Estéreo'})")
    return str(output_wav_path)


def batch_extract_audio(
    input_dir: str,
    output_dir: str = None,
    extensions: list = None,
    **kwargs,
) -> list:
    """
    Extrae el audio de TODOS los videos en un directorio dado.

    Args:
        input_dir (str):
            Directorio que contiene los archivos de video.

        output_dir (str, optional):
            Directorio donde se guardarán los archivos .wav.
            Si no se especifica, se guardan junto a cada video original.

        extensions (list, optional):
            Lista de extensiones de video a buscar.
            Por defecto: ['.mp4', '.mkv', '.avi', '.mov'].

        **kwargs:
            Parámetros adicionales pasados a `extract_audio_from_video`
            (ej. sample_rate=44100, channels=2).

    Returns:
        list: Lista de rutas absolutas de los archivos .wav generados.

    Ejemplo de uso:
        >>> wavs = batch_extract_audio("videos/", "audios/", sample_rate=22050)
        >>> print(f"Se procesaron {len(wavs)} archivos.")
    """
    if extensions is None:
        extensions = [".mp4", ".mkv", ".avi", ".mov"]

    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"El directorio no existe: {input_dir}")

    # Buscamos todos los archivos de video con las extensiones especificadas
    video_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in extensions
    ]

    if not video_files:
        print(f"⚠️  No se encontraron videos con extensiones {extensions} en: {input_dir}")
        return []

    print(f"📁 Se encontraron {len(video_files)} video(s) a procesar.\n")
    output_paths = []

    for video_file in sorted(video_files):
        # Construimos la ruta de salida del .wav
        if output_dir:
            out_path = Path(output_dir) / video_file.with_suffix(".wav").name
        else:
            out_path = None  # extract_audio_from_video lo coloca junto al video

        wav_path = extract_audio_from_video(
            video_path=str(video_file),
            output_wav_path=str(out_path) if out_path else None,
            **kwargs,
        )
        output_paths.append(wav_path)
        print()  # Línea en blanco entre archivos para mejorar legibilidad

    print(f"🎉 Proceso completo. {len(output_paths)} archivos .wav generados.")
    return output_paths
