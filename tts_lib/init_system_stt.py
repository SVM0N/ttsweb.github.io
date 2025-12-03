"""STT system initialization.

This module handles initialization of:
- STT backend (Whisper or Faster-Whisper)
- Configuration (device, output directory)
"""

from pathlib import Path
from typing import Tuple, Optional
from .stt_backends import get_stt_backend
from .config import TTSConfig  # Reuse TTSConfig since it's generic enough


def initialize_system(
    stt_model: str,
    output_dir: str = "files",
    device: str = "auto"
) -> Tuple:
    """Initialize STT system with model and configuration.

    Args:
        stt_model: STT model name (e.g., "whisper-base", "faster-whisper-small")
        output_dir: Output directory for transcripts
        device: Device to use ("auto", "cuda", "cpu", "mps")

    Returns:
        Tuple of (stt_backend, config)
    """
    print("="*60)
    print("INITIALIZING STT SYSTEM")
    print("="*60)

    # Initialize config (reusing TTSConfig as it's generic)
    print("\n📋 Setting up configuration...")
    config = TTSConfig(output_dir=output_dir, device=device)

    # Get actual device (after auto-detection)
    actual_device = config.device

    # Initialize STT backend
    print(f"\n🎤 Loading STT model: {stt_model}")
    stt = get_stt_backend(stt_model, device=actual_device)

    print("\n" + "="*60)
    print("✓ STT SYSTEM READY")
    print("="*60)

    return stt, config
