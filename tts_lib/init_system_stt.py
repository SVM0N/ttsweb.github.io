"""STT system initialization.

This module handles initialization of:
- STT backend (Whisper, Faster-Whisper, or WhisperX)
- Configuration (device, output directory)
"""

import os
from pathlib import Path
from typing import Tuple, Optional
from .stt_backends import get_stt_backend
from .config import TTSConfig  # Reuse TTSConfig since it's generic enough


def initialize_system(
    stt_model: str,
    output_dir: str = "files",
    device: str = "auto",
    hf_token: Optional[str] = None
) -> Tuple:
    """Initialize STT system with model and configuration.

    Args:
        stt_model: STT model name (e.g., "whisper-base", "faster-whisper-small", "whisperx-base")
        output_dir: Output directory for transcripts
        device: Device to use ("auto", "cuda", "cpu", "mps")
        hf_token: HuggingFace token (required for WhisperX diarization)

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

    # Check for HF token in environment if not provided
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN')

    # Initialize STT backend
    print(f"\n🎤 Loading STT model: {stt_model}")
    stt = get_stt_backend(stt_model, device=actual_device, hf_token=hf_token)

    # Warn if WhisperX but no token
    if stt_model.startswith("whisperx-") and not hf_token:
        print("\n⚠️  WhisperX detected but no HuggingFace token provided")
        print("   Speaker diarization will be disabled")
        print("   To enable: Set HF_TOKEN environment variable or pass hf_token parameter")
        print("   Get token from: https://huggingface.co/settings/tokens")

    print("\n" + "="*60)
    print("✓ STT SYSTEM READY")
    print("="*60)

    return stt, config
