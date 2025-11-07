"""
TTS Conversion Examples

This module contains example usage functions for different conversion types:
- String to audio
- PDF to audio
- EPUB to per-chapter audio ZIP
"""

import os
from pathlib import Path


def run_conversion(conversion_type, tts, config, pdf_extractor, tts_model,
                   out_format="wav",
                   pdf_path="files/Case1Writeup.pdf", pdf_pages=None,
                   epub_path="book.epub", zip_name=""):
    """
    Universal conversion function that routes to the appropriate conversion type.

    Args:
        conversion_type: One of "string", "pdf", or "epub"
        tts: TTS backend instance
        config: TTSConfig instance
        pdf_extractor: PDF extractor instance (only needed for PDF conversion)
        tts_model: Name of the TTS model being used
        out_format: Output format ("wav" or "mp3")
        pdf_path: Path to PDF file (for PDF conversion)
        pdf_pages: List of page numbers or None for all pages (for PDF conversion)
        epub_path: Path to EPUB file (for EPUB conversion)
        zip_name: Custom ZIP name (for EPUB conversion)

    Returns:
        Tuple of (audio_path, manifest_path) for string/pdf, or zip_path for epub
    """
    # Auto-detect if running in Google Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    print("="*60)
    print(f"RUNNING CONVERSION: {conversion_type.upper()}")
    print("="*60)
    print()

    if conversion_type == "string":
        return run_string_to_audio(
            tts=tts,
            config=config,
            tts_model=tts_model,
            out_format=out_format,
            in_colab=in_colab
        )

    elif conversion_type == "pdf":
        return run_pdf_to_audio(
            tts=tts,
            config=config,
            pdf_extractor=pdf_extractor,
            tts_model=tts_model,
            out_format=out_format,
            pdf_path=pdf_path,
            pages=pdf_pages,
            in_colab=in_colab
        )

    elif conversion_type == "epub":
        return run_epub_to_audio(
            tts=tts,
            config=config,
            tts_model=tts_model,
            out_format=out_format,
            epub_path=epub_path,
            zip_name=zip_name,
            in_colab=in_colab
        )

    else:
        print(f"⚠️  Invalid conversion type: {conversion_type}")
        print("   Valid options: 'string', 'pdf', 'epub'")
        return None


def run_string_to_audio(tts, config, tts_model, out_format="wav", in_colab=False):
    """Convert text string to audio."""
    from tts_lib.synthesis import synth_string

    # Configuration
    VOICE = None  # Use default voice (or specify a voice description)
    SPEED = 1.0   # Speech speed (Kokoro and Maya1 only)

    # Text to synthesize
    # For Maya1: You can add emotion tags like <laugh>, <whisper>, <cry>, etc.
    TEXT = """Hello! This is a test of the unified TTS system.
    It automatically installs only the dependencies you need.
    """

    # Run synthesis
    audio_path, manifest_path = synth_string(
        tts=tts,
        config=config,
        text=TEXT,
        voice=VOICE,
        speed=SPEED,
        out_format=out_format,
        tts_model=tts_model
    )

    print(f"\n✓ Audio saved to: {audio_path}")
    print(f"✓ Manifest saved to: {manifest_path}")

    if in_colab:
        print("\n💡 To download the files, run:")
        print(f"   from google.colab import files")
        print(f"   files.download('{audio_path}')")
        print(f"   files.download('{manifest_path}')")

    return audio_path, manifest_path


def run_pdf_to_audio(tts, config, pdf_extractor, tts_model, out_format="wav",
                     pdf_path="files/Case1Writeup.pdf", pages=None, in_colab=False):
    """Convert PDF to audio with optional page selection."""
    from tts_lib.synthesis import synth_pdf

    # Configuration
    VOICE = None  # Use default voice
    SPEED = 1.0

    # Check if file exists, provide helpful message if not
    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF file not found: {pdf_path}")
        if in_colab:
            print("\n📤 To upload a PDF in Colab, run:")
            print("   from google.colab import files")
            print("   uploaded = files.upload()")
            print("   # Then move it: !mv uploaded_file.pdf files/")
        else:
            print("\n💡 Make sure your PDF is in the 'files' directory")
            print("   Or update the pdf_path parameter")
        return None, None

    # Run synthesis
    audio_path, manifest_path = synth_pdf(
        tts=tts,
        config=config,
        pdf_extractor=pdf_extractor,
        file_path_or_bytes=pdf_path,
        voice=VOICE,
        speed=SPEED,
        out_format=out_format,
        pages=pages,
        tts_model=tts_model
    )

    print(f"\n✓ Audio saved to: {audio_path}")
    print(f"✓ Manifest saved to: {manifest_path}")

    if in_colab:
        print("\n💡 To download the files, run:")
        print(f"   from google.colab import files")
        print(f"   files.download('{audio_path}')")
        print(f"   files.download('{manifest_path}')")

    return audio_path, manifest_path


def run_epub_to_audio(tts, config, tts_model, out_format="wav",
                      epub_path="book.epub", zip_name="", in_colab=False):
    """Convert EPUB to per-chapter audio ZIP."""
    from tts_lib.synthesis import synth_epub

    # Configuration
    VOICE = None  # Use default voice
    SPEED = 1.0

    # Check if file exists, provide helpful message if not
    if not os.path.exists(epub_path):
        print(f"⚠️  EPUB file not found: {epub_path}")
        if in_colab:
            print("\n📤 To upload an EPUB in Colab, run:")
            print("   from google.colab import files")
            print("   uploaded = files.upload()")
        else:
            print("\n💡 Make sure your EPUB file exists")
            print("   Or update the epub_path parameter")
        return None

    # Run synthesis
    zip_path = synth_epub(
        tts=tts,
        config=config,
        file_path_or_bytes=epub_path,
        voice=VOICE,
        speed=SPEED,
        per_chapter_format=out_format,
        zip_name=(zip_name or None),
        tts_model=tts_model
    )

    print(f"\n✓ ZIP archive saved to: {zip_path}")

    if in_colab:
        print("\n💡 To download the ZIP file, run:")
        print(f"   from google.colab import files")
        print(f"   files.download('{zip_path}')")

    return zip_path
