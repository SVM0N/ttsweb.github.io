"""High-level synthesis functions for TTS processing.

This module provides user-friendly wrapper functions for:
- Text string synthesis
- PDF synthesis
- EPUB synthesis

All functions handle model-specific parameters automatically.
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Union, Optional, List, Tuple

from .tts_utils import wav_to_mp3_bytes, safe_name, extract_chapters_from_epub
from .manifest import create_manifest, save_manifest


def synth_string(
    tts,
    config,
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    out_format: str = "wav",
    basename: str = "tts_text",
    tts_model: str = "kokoro_1.0",
    enable_text_input: bool = True,
    enable_mp3_output: bool = True,
    **kwargs
) -> Tuple[str, str]:
    """Synthesize a text string to audio.

    Args:
        tts: TTS backend instance
        config: TTSConfig instance
        text: Text to synthesize
        voice: Voice/speaker to use (None for default)
        speed: Speech speed (Kokoro only)
        out_format: Output format ("wav" or "mp3")
        basename: Base name for output files
        tts_model: TTS model name
        enable_text_input: Whether text input is enabled
        enable_mp3_output: Whether MP3 output is enabled
        **kwargs: Additional model-specific parameters

    Returns:
        Tuple of (audio_path, manifest_path)
    """
    if not enable_text_input:
        raise ValueError("Text input is disabled. Set ENABLE_TEXT_INPUT=True in configuration.")

    if out_format == "mp3" and not enable_mp3_output:
        raise ValueError("MP3 output is disabled. Set ENABLE_MP3_OUTPUT=True in configuration.")

    voice = voice or tts.get_default_voice()

    elements = [{
        "text": text,
        "metadata": {"page_number": 1, "source": "string", "points": None}
    }]

    # Synthesize based on model type
    if tts_model.startswith("kokoro"):
        wav_bytes, timeline = tts.synthesize_text_to_wav(
            elements, voice=voice, speed=speed, **kwargs
        )
    elif tts_model == "maya1":
        wav_bytes, timeline = tts.synthesize_text_to_wav(
            elements, voice=voice, **kwargs
        )
    else:  # Silero
        wav_bytes, timeline = tts.synthesize_text_to_wav(
            elements, speaker=voice, **kwargs
        )

    # Save audio
    out_base = config.get_output_path(basename)

    if out_format.lower() == "mp3":
        mp3 = wav_to_mp3_bytes(wav_bytes)
        audio_path = str(out_base) + ".mp3"
        with open(audio_path, "wb") as f:
            f.write(mp3)
    else:
        audio_path = str(out_base) + ".wav"
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)

    # Save manifest
    manifest_path = str(out_base) + "_manifest.json"
    manifest = create_manifest(Path(audio_path).name, timeline)
    save_manifest(manifest, manifest_path)

    return audio_path, manifest_path


def synth_pdf(
    tts,
    config,
    pdf_extractor,
    file_path_or_bytes: Union[str, Path, io.BytesIO],
    voice: Optional[str] = None,
    speed: float = 1.0,
    out_format: str = "wav",
    basename: Optional[str] = None,
    pages: Optional[List[int]] = None,
    tts_model: str = "kokoro_1.0",
    enable_pdf_input: bool = True,
    enable_mp3_output: bool = True,
    **kwargs
) -> Tuple[str, str]:
    """Synthesize a PDF to audio.

    Args:
        tts: TTS backend instance
        config: TTSConfig instance
        pdf_extractor: PDF extractor instance
        file_path_or_bytes: PDF file path or BytesIO object
        voice: Voice/speaker to use (None for default)
        speed: Speech speed (Kokoro only)
        out_format: Output format ("wav" or "mp3")
        basename: Base name for output files (None for auto from filename)
        pages: List of page numbers to extract (None for all pages)
        tts_model: TTS model name
        enable_pdf_input: Whether PDF input is enabled
        enable_mp3_output: Whether MP3 output is enabled
        **kwargs: Additional model-specific parameters

    Returns:
        Tuple of (audio_path, manifest_path)
    """
    if not enable_pdf_input:
        raise ValueError("PDF input is disabled. Set ENABLE_PDF_INPUT=True in configuration.")

    if pdf_extractor is None:
        raise ValueError("No PDF extractor configured. Set PDF_EXTRACTOR in configuration.")

    if out_format == "mp3" and not enable_mp3_output:
        raise ValueError("MP3 output is disabled. Set ENABLE_MP3_OUTPUT=True in configuration.")

    voice = voice or tts.get_default_voice()

    # Load PDF
    if isinstance(file_path_or_bytes, (str, Path)):
        with open(file_path_or_bytes, "rb") as fh:
            pdf_bytes = io.BytesIO(fh.read())
        stem = Path(file_path_or_bytes).stem
    else:
        pdf_bytes = file_path_or_bytes
        stem = basename or "document"

    # Extract text
    elements = pdf_extractor.extract(pdf_bytes, pages=pages)

    # Synthesize based on model type
    if tts_model.startswith("kokoro"):
        wav_bytes, timeline = tts.synthesize_text_to_wav(
            elements, voice=voice, speed=speed, **kwargs
        )
    elif tts_model == "maya1":
        wav_bytes, timeline = tts.synthesize_text_to_wav(
            elements, voice=voice, **kwargs
        )
    else:  # Silero
        wav_bytes, timeline = tts.synthesize_text_to_wav(
            elements, speaker=voice, **kwargs
        )

    # Save audio
    out_base = config.get_output_path(f"{basename or stem}_tts")

    if out_format.lower() == "mp3":
        mp3 = wav_to_mp3_bytes(wav_bytes)
        audio_path = str(out_base) + ".mp3"
        with open(audio_path, "wb") as f:
            f.write(mp3)
    else:
        audio_path = str(out_base) + ".wav"
        with open(audio_path, "wb") as f:
            f.write(wav_bytes)

    # Save manifest
    manifest_path = str(out_base) + "_manifest.json"
    manifest = create_manifest(Path(audio_path).name, timeline)
    save_manifest(manifest, manifest_path)

    return audio_path, manifest_path


def synth_epub(
    tts,
    config,
    file_path_or_bytes: Union[str, Path, io.BytesIO],
    voice: Optional[str] = None,
    speed: float = 1.0,
    per_chapter_format: str = "wav",
    zip_name: Optional[str] = None,
    tts_model: str = "kokoro_1.0",
    enable_epub_input: bool = True,
    enable_mp3_output: bool = True,
    **kwargs
) -> str:
    """Synthesize an EPUB to per-chapter audio files in a ZIP.

    Args:
        tts: TTS backend instance
        config: TTSConfig instance
        file_path_or_bytes: EPUB file path or BytesIO object
        voice: Voice/speaker to use (None for default)
        speed: Speech speed (Kokoro only)
        per_chapter_format: Output format per chapter ("wav" or "mp3")
        zip_name: Name for output ZIP file (None for auto from filename)
        tts_model: TTS model name
        enable_epub_input: Whether EPUB input is enabled
        enable_mp3_output: Whether MP3 output is enabled
        **kwargs: Additional model-specific parameters

    Returns:
        Path to output ZIP file
    """
    if not enable_epub_input:
        raise ValueError("EPUB input is disabled. Set ENABLE_EPUB_INPUT=True in configuration.")

    if per_chapter_format == "mp3" and not enable_mp3_output:
        raise ValueError("MP3 output is disabled. Set ENABLE_MP3_OUTPUT=True in configuration.")

    voice = voice or tts.get_default_voice()

    # Load EPUB
    if isinstance(file_path_or_bytes, (str, Path)):
        with open(file_path_or_bytes, "rb") as fh:
            epub_bytes = io.BytesIO(fh.read())
        stem = Path(file_path_or_bytes).stem
    else:
        epub_bytes = file_path_or_bytes
        stem = "book"

    # Extract chapters
    chapters = extract_chapters_from_epub(epub_bytes)
    assert chapters, "No chapters detected in EPUB."

    # Create ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (title, body) in enumerate(chapters, 1):
            name = f"{idx:02d}_{safe_name(title)[:40]}"

            chapter_elements = [{
                "text": body,
                "metadata": {
                    "chapter_index": idx,
                    "chapter_title": title,
                    "page_number": 1,
                    "points": None
                }
            }]

            # Synthesize chapter based on model type
            if tts_model.startswith("kokoro"):
                wav_bytes, timeline = tts.synthesize_text_to_wav(
                    chapter_elements, voice=voice, speed=speed, **kwargs
                )
            elif tts_model == "maya1":
                wav_bytes, timeline = tts.synthesize_text_to_wav(
                    chapter_elements, voice=voice, **kwargs
                )
            else:  # Silero
                wav_bytes, timeline = tts.synthesize_text_to_wav(
                    chapter_elements, speaker=voice, **kwargs
                )

            # Add audio to ZIP
            if per_chapter_format.lower() == "mp3":
                data = wav_to_mp3_bytes(wav_bytes)
                audio_name = f"{name}.mp3"
                zf.writestr(audio_name, data)
            else:
                audio_name = f"{name}.wav"
                zf.writestr(audio_name, wav_bytes)

            # Add manifest to ZIP
            manifest = create_manifest(audio_name, timeline)
            zf.writestr(f"{name}_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    # Save ZIP
    zip_buf.seek(0)
    zpath = str(config.get_output_path(f"{zip_name or (stem + '_chapters')}.zip"))
    with open(zpath, "wb") as f:
        f.write(zip_buf.read())

    return zpath
