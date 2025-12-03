"""Automatic dependency installation for STT notebook.

This module handles smart dependency installation based on user configuration.
Only installs the packages actually needed for the selected STT model and output formats.
"""

import subprocess
import sys


def install_package(package):
    """Install a package using pip."""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    print(f"✓ {package} installed")


def install_dependencies(
    stt_model: str,
    output_formats: dict
):
    """Install only the dependencies needed for the selected configuration.

    Args:
        stt_model: STT model name (e.g., "whisper-base", "faster-whisper-small")
        output_formats: Dictionary of output formats (txt, srt, vtt, json)
    """
    print("="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)

    # Core dependencies (always needed)
    print("\n📦 Installing core dependencies...")

    # Install PyTorch
    try:
        import torch
        print(f"✓ torch already installed")
    except ImportError:
        # Detect if running in Colab (likely has GPU)
        try:
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False

        if in_colab:
            print("Installing torch with CUDA support for Colab...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ])
            print("✓ torch installed with CUDA support")
        else:
            print("Installing torch (auto-detect CUDA)...")
            install_package("torch")

    # Install NumPy
    try:
        import numpy
        print(f"✓ numpy already installed")
    except ImportError:
        install_package("numpy")

    # Note about ffmpeg for audio processing
    print("\n⚠️  NOTE: Audio processing requires ffmpeg to be installed on your system:")
    print("   - macOS: brew install ffmpeg")
    print("   - Linux: sudo apt-get install ffmpeg")
    print("   - Windows: Download from https://ffmpeg.org/")
    print("   - Colab: ffmpeg is pre-installed ✓")

    # STT model dependencies
    print("\n🎤 Installing STT model dependencies...")

    if stt_model.startswith("whisperx-"):
        # WhisperX with diarization
        try:
            import whisperx
            print(f"✓ whisperx already installed")
        except ImportError:
            install_package("git+https://github.com/m-bain/whisperx.git")

        # Install pyannote for speaker diarization
        try:
            import pyannote.audio
            print(f"✓ pyannote.audio already installed")
        except ImportError:
            install_package("pyannote.audio")

        print("\n📝 NOTE: WhisperX provides:")
        print("   - Word-level timestamps (more precise than Whisper)")
        print("   - Speaker diarization (separates different speakers)")
        print("   - Requires HuggingFace token for diarization")
        print("   - Get token: https://huggingface.co/settings/tokens")
        print("   - Accept pyannote terms: https://huggingface.co/pyannote/speaker-diarization")

    elif stt_model.startswith("whisper-"):
        # Standard OpenAI Whisper
        try:
            import whisper
            print(f"✓ openai-whisper already installed")
        except ImportError:
            install_package("openai-whisper")

    elif stt_model.startswith("faster-whisper-"):
        # Faster-Whisper (optimized)
        try:
            import faster_whisper
            print(f"✓ faster-whisper already installed")
        except ImportError:
            install_package("faster-whisper")

        print("\n📝 NOTE: Faster-Whisper provides:")
        print("   - ~4x faster transcription than standard Whisper")
        print("   - Lower memory usage")
        print("   - Same accuracy as standard Whisper")
        print("   - Does NOT support MPS (Apple Silicon GPU), will use CPU")

    # Output format dependencies (mostly built-in, but check JSON)
    if output_formats.get('json', False):
        # json is built-in, no need to install
        pass

    print("\n" + "="*60)
    print("✓ ALL DEPENDENCIES INSTALLED")
    print("="*60)
