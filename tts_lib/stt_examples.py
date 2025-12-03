"""Example STT workflows and convenience functions.

This module provides high-level functions for common STT tasks.
"""

from pathlib import Path
from typing import Dict, Optional
from .output_formatters import save_transcription


def run_transcription(
    stt,
    config,
    audio_path: str,
    output_formats: Dict[str, bool],
    language: Optional[str] = None,
    task: str = "transcribe",
    enable_diarization: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> Dict:
    """Run complete transcription workflow.

    Args:
        stt: STT backend instance
        config: Configuration instance
        audio_path: Path to audio file
        output_formats: Dictionary of output formats to generate
        language: Language code (None for auto-detect)
        task: "transcribe" or "translate"
        enable_diarization: Enable speaker diarization (WhisperX only)
        min_speakers: Minimum number of speakers (for diarization)
        max_speakers: Maximum number of speakers (for diarization)

    Returns:
        Dictionary with results and output file paths
    """
    print("="*60)
    print("RUNNING TRANSCRIPTION")
    print("="*60)

    # Validate audio file exists
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"\n📁 Audio file: {audio_path}")
    print(f"📝 Language: {language or 'Auto-detect'}")
    print(f"🎯 Task: {task}")

    # Run transcription
    print("\n🎤 Transcribing audio...")

    # Check if backend supports diarization
    if hasattr(stt, 'transcribe') and 'enable_diarization' in stt.transcribe.__code__.co_varnames:
        # WhisperX backend - supports diarization
        result = stt.transcribe(
            audio_path=audio_path,
            language=language,
            task=task,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    else:
        # Standard Whisper or Faster-Whisper - no diarization
        if enable_diarization:
            print("⚠️  Speaker diarization requested but not supported by this model")
            print("   Use WhisperX models for speaker diarization")
        result = stt.transcribe(
            audio_path=audio_path,
            language=language,
            task=task
        )

    # Generate base output path (without extension)
    base_output_name = audio_path.stem + "_transcript"
    base_output_path = config.get_output_path(base_output_name)

    # Save in requested formats
    print(f"\n💾 Saving transcription...")
    output_files = save_transcription(
        result=result,
        base_path=base_output_path,
        output_formats=output_formats
    )

    print("\n" + "="*60)
    print("✓ TRANSCRIPTION COMPLETE")
    print("="*60)

    return {
        "text": result["text"],
        "segments": result["segments"],
        "language": result.get("language", language),
        "output_files": output_files
    }
