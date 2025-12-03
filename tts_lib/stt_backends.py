"""STT backend implementations for various speech-to-text models.

This module provides backend classes for:
- OpenAI Whisper (tiny, base, small, medium, large)
- Faster-Whisper (optimized versions)
- WhisperX (with speaker diarization and word-level timestamps)

Supports both audio and video files (MP4, MOV, AVI, etc.) by extracting audio.
"""

import torch
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Union


def extract_audio_from_video(video_path: Union[str, Path]) -> Path:
    """Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file

    Returns:
        Path to temporary audio file (WAV format)

    Raises:
        RuntimeError: If ffmpeg is not available or extraction fails
    """
    video_path = Path(video_path)

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg is required to process video files. Install it with:\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: sudo apt-get install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/"
        )

    # Create temporary WAV file
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio_path = Path(temp_audio.name)
    temp_audio.close()

    print(f"Extracting audio from video: {video_path.name}")

    # Extract audio using ffmpeg
    try:
        subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # WAV format
            '-ar', '16000',  # 16kHz sample rate (Whisper's native)
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            str(temp_audio_path)
        ], check=True, capture_output=True)

        print(f"✓ Audio extracted to temporary file")
        return temp_audio_path

    except subprocess.CalledProcessError as e:
        temp_audio_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")


def is_video_file(file_path: Union[str, Path]) -> bool:
    """Check if file is a video file based on extension.

    Args:
        file_path: Path to file

    Returns:
        True if file appears to be a video file
    """
    video_extensions = {
        '.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm',
        '.m4v', '.mpg', '.mpeg', '.3gp', '.ogv'
    }
    return Path(file_path).suffix.lower() in video_extensions


class WhisperBackend:
    """Backend for OpenAI Whisper models."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """Initialize Whisper backend.

        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            device: Device to use ("cuda", "cpu", "mps")
        """
        import whisper

        self.model_size = model_size
        self.device = device

        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size, device=device)
        print(f"✓ Whisper {model_size} loaded on {device}")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """Transcribe audio or video file.

        Args:
            audio_path: Path to audio or video file
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate"

        Returns:
            Dictionary with transcription results including:
            - text: Full transcript
            - segments: List of segments with timestamps
            - language: Detected language
        """
        audio_path = Path(audio_path)
        temp_audio_path = None

        # Check if input is a video file
        if is_video_file(audio_path):
            temp_audio_path = extract_audio_from_video(audio_path)
            actual_audio_path = temp_audio_path
        else:
            actual_audio_path = audio_path

        try:
            print(f"Transcribing {audio_path.name}...")

            options = {
                "task": task,
                "fp16": self.device == "cuda",  # Use FP16 only on CUDA
            }

            if language:
                options["language"] = language

            result = self.model.transcribe(str(actual_audio_path), **options)

            print(f"✓ Transcription complete")
            return result

        finally:
            # Clean up temporary audio file
            if temp_audio_path and temp_audio_path.exists():
                temp_audio_path.unlink()
                print(f"✓ Cleaned up temporary audio file")


class FasterWhisperBackend:
    """Backend for Faster-Whisper (optimized) models."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """Initialize Faster-Whisper backend.

        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            device: Device to use ("cuda" or "cpu" - MPS not supported)
        """
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.device = device

        # Faster-Whisper doesn't support MPS, fall back to CPU
        if device == "mps":
            print("⚠️  Faster-Whisper doesn't support MPS, falling back to CPU")
            device = "cpu"
            self.device = "cpu"

        # Determine compute type based on device
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"

        print(f"Loading Faster-Whisper {model_size} model...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        print(f"✓ Faster-Whisper {model_size} loaded on {device}")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """Transcribe audio or video file.

        Args:
            audio_path: Path to audio or video file
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate"

        Returns:
            Dictionary with transcription results including:
            - text: Full transcript
            - segments: List of segments with timestamps
            - language: Detected language
        """
        audio_path = Path(audio_path)
        temp_audio_path = None

        # Check if input is a video file
        if is_video_file(audio_path):
            temp_audio_path = extract_audio_from_video(audio_path)
            actual_audio_path = temp_audio_path
        else:
            actual_audio_path = audio_path

        try:
            print(f"Transcribing {audio_path.name}...")

            # Transcribe
            segments, info = self.model.transcribe(
                str(actual_audio_path),
                language=language,
                task=task
            )

            # Convert segments iterator to list and build result
            segments_list = []
            full_text = []

            for segment in segments:
                segments_list.append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob,
                })
                full_text.append(segment.text)

            result = {
                "text": " ".join(full_text),
                "segments": segments_list,
                "language": info.language if hasattr(info, 'language') else language,
            }

            print(f"✓ Transcription complete")
            return result

        finally:
            # Clean up temporary audio file
            if temp_audio_path and temp_audio_path.exists():
                temp_audio_path.unlink()
                print(f"✓ Cleaned up temporary audio file")


class WhisperXBackend:
    """Backend for WhisperX with speaker diarization and word-level timestamps."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        hf_token: Optional[str] = None
    ):
        """Initialize WhisperX backend.

        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large-v2")
            device: Device to use ("cuda" or "cpu")
            compute_type: Compute type ("int8", "float16", "float32")
            hf_token: HuggingFace token for pyannote speaker diarization model
        """
        import whisperx

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.hf_token = hf_token

        print(f"Loading WhisperX {model_size} model...")
        self.model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type
        )
        print(f"✓ WhisperX {model_size} loaded on {device}")

        # Will be loaded on-demand for alignment and diarization
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        enable_diarization: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict:
        """Transcribe audio or video file with optional speaker diarization.

        Args:
            audio_path: Path to audio or video file
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate"
            enable_diarization: Enable speaker diarization
            min_speakers: Minimum number of speakers (for diarization)
            max_speakers: Maximum number of speakers (for diarization)

        Returns:
            Dictionary with transcription results including:
            - text: Full transcript
            - segments: List of segments with timestamps
            - language: Detected language
            - word_segments: Word-level timestamps (if aligned)
            - speakers: Speaker labels (if diarization enabled)
        """
        import whisperx

        audio_path = Path(audio_path)
        temp_audio_path = None

        # Check if input is a video file
        if is_video_file(audio_path):
            temp_audio_path = extract_audio_from_video(audio_path)
            actual_audio_path = temp_audio_path
        else:
            actual_audio_path = audio_path

        try:
            print(f"Transcribing with WhisperX: {audio_path.name}...")

            # Load audio
            audio = whisperx.load_audio(str(actual_audio_path))

            # 1. Transcribe with WhisperX
            result = self.model.transcribe(
                audio,
                language=language,
                task=task
            )

            detected_language = result.get("language", language)
            print(f"✓ Transcription complete (language: {detected_language})")

            # 2. Align whisper output for word-level timestamps
            if not self.align_model or self.align_metadata.get("language") != detected_language:
                print(f"Loading alignment model for {detected_language}...")
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device
                )

            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            print(f"✓ Alignment complete (word-level timestamps)")

            # 3. Assign speaker labels (if enabled)
            if enable_diarization:
                if not self.hf_token:
                    print("⚠️  WARNING: No HuggingFace token provided for diarization")
                    print("   Set HF_TOKEN environment variable or pass hf_token parameter")
                    print("   Get token from: https://huggingface.co/settings/tokens")
                    print("   Skipping diarization...")
                else:
                    if not self.diarize_model:
                        print("Loading speaker diarization model...")
                        self.diarize_model = whisperx.DiarizationPipeline(
                            use_auth_token=self.hf_token,
                            device=self.device
                        )

                    print("Running speaker diarization...")
                    diarize_segments = self.diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )

                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    print(f"✓ Speaker diarization complete")

            # Format result
            segments_list = result.get("segments", [])
            full_text = " ".join([seg.get("text", "") for seg in segments_list])

            formatted_result = {
                "text": full_text,
                "segments": segments_list,
                "language": detected_language,
                "word_segments": result.get("word_segments", []),
            }

            return formatted_result

        finally:
            # Clean up temporary audio file
            if temp_audio_path and temp_audio_path.exists():
                temp_audio_path.unlink()
                print(f"✓ Cleaned up temporary audio file")


def get_stt_backend(
    model_name: str,
    device: str = "cpu",
    hf_token: Optional[str] = None
):
    """Factory function to get the appropriate STT backend.

    Args:
        model_name: Model name (e.g., "whisper-base", "faster-whisper-small", "whisperx-base")
        device: Device to use ("cuda", "cpu", "mps")
        hf_token: HuggingFace token (required for WhisperX diarization)

    Returns:
        STT backend instance

    Raises:
        ValueError: If model name is not recognized
    """
    if model_name.startswith("whisperx-"):
        # WhisperX models with diarization
        model_size = model_name.replace("whisperx-", "")
        # Determine compute type based on device
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"
        return WhisperXBackend(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            hf_token=hf_token
        )

    elif model_name.startswith("whisper-"):
        # Standard Whisper models
        model_size = model_name.replace("whisper-", "")
        return WhisperBackend(model_size=model_size, device=device)

    elif model_name.startswith("faster-whisper-"):
        # Faster-Whisper models
        model_size = model_name.replace("faster-whisper-", "")
        return FasterWhisperBackend(model_size=model_size, device=device)

    else:
        raise ValueError(
            f"Unknown STT model: {model_name}. "
            f"Expected format: 'whisper-<size>', 'faster-whisper-<size>', or 'whisperx-<size>'"
        )
