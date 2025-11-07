"""Common utilities for TTS processing.

This module contains shared functionality used across all TTS backends:
- EPUB extraction
- Sentence splitting
- WAV to MP3 conversion
- File naming utilities
"""

import io
import re
import zipfile
from pathlib import Path
from typing import List, Tuple
from ebooklib import epub
from pydub import AudioSegment


# Sentence splitting pattern - keeps chunks small to avoid phoneme truncation
SPLIT_PATTERN = r"[.?!]\s+|[\n]{2,}"
SPLIT_PATTERN_CAP = r"([.?!]\s+|[\n]{2,})"


def extract_chapters_from_epub(file_like: io.BytesIO) -> List[Tuple[str, str]]:
    """Extract chapters from an EPUB file.

    Args:
        file_like: EPUB file as BytesIO object

    Returns:
        List of (title, text) tuples for each chapter
    """
    bk = epub.read_epub(file_like)
    chapters = []

    for item in bk.get_items_of_type(epub.ITEM_DOCUMENT):
        if getattr(item, "is_nav", False):
            continue

        html = item.get_content().decode("utf-8", errors="ignore")

        # Clean HTML
        text = re.sub(r"<(script|style).*?>.*?</\1>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</p>|</div>|</h\d>", "\n\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if text:
            title = Path(item.file_name).stem
            first = text.splitlines()[0] if text else ""
            m = re.match(r"(?i)\s*(chapter|part|book)\b[^\n]{0,80}", first)
            if m:
                title = first[:60]
            chapters.append((title, text))

    # Fallback if no chapters found
    if not chapters:
        blobs = []
        for item in bk.get_items_of_type(epub.ITEM_DOCUMENT):
            if getattr(item, "is_nav", False):
                continue
            blobs.append(item.get_content().decode("utf-8", errors="ignore"))

        html = " ".join(blobs)
        text = re.sub(r"<(script|style).*?>.*?</\1>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</p>|</div>|</h\d>", "\n\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            chapters = [("Chapter 1", text)]

    return chapters


def split_sentences_keep_delim(text: str) -> List[str]:
    """Split text into sentences while keeping delimiters.

    Splits on periods, question marks, exclamation points, and double newlines.
    Keeps the delimiter attached to the sentence.

    Args:
        text: Text to split

    Returns:
        List of sentences with delimiters attached
    """
    parts = re.split(SPLIT_PATTERN_CAP, text)
    sents = []

    for i in range(0, len(parts), 2):
        chunk = (parts[i] or "").strip()
        sep = parts[i + 1] if i + 1 < len(parts) else ""

        if not chunk:
            continue

        if sep and not sep.isspace():
            chunk = (chunk + " " + sep.strip()).strip()

        sents.append(chunk)

    return sents


def wav_to_mp3_bytes(wav_bytes: bytes, bitrate: str = "128k") -> bytes:
    """Convert WAV bytes to MP3 bytes.

    Args:
        wav_bytes: WAV audio data
        bitrate: MP3 bitrate (default: "128k")

    Returns:
        MP3 audio data as bytes
    """
    audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    out = io.BytesIO()
    audio.export(out, format="mp3", bitrate=bitrate)
    out.seek(0)
    return out.read()


def safe_name(s: str) -> str:
    """Convert a string to a safe filename.

    Args:
        s: String to sanitize

    Returns:
        Sanitized string suitable for use as a filename
    """
    s = re.sub(r"[^\w\-]+", "_", s).strip("_")
    return s or "chapter"
