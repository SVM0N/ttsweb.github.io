"""Configuration management for TTS system.

This module provides configuration utilities for:
- Device selection (CUDA/CPU/MPS)
- Output directory management
- Model parameters
"""

import os
import torch
from pathlib import Path
from typing import Union, Optional


class TTSConfig:
    """Configuration manager for TTS system."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        """Initialize TTS configuration.

        Args:
            output_dir: Output directory for generated files (default: current directory)
            device: Device to use ("auto", "cuda", "cpu", "mps", or torch.device)
        """
        self.output_dir = self._setup_output_dir(output_dir)
        self.device = self._setup_device(device)

    def _setup_output_dir(self, output_dir: Optional[Union[str, Path]]) -> Path:
        """Set up and validate output directory.

        Args:
            output_dir: Output directory path or None for current directory

        Returns:
            Resolved Path object
        """
        if output_dir is None:
            output_dir = Path(".")
        else:
            output_dir = Path(output_dir)

        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved = output_dir.resolve()
        print(f"Output directory: {resolved}")

        return resolved

    def _setup_device(self, device: Optional[Union[str, torch.device]]) -> Union[str, torch.device]:
        """Set up and validate device.

        Args:
            device: Device specification or None for auto-detection

        Returns:
            Device string or torch.device object
        """
        if device is None or device == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                selected = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Check if we're on Apple Silicon
                import platform
                is_apple_silicon = (
                    platform.system() == "Darwin" and
                    platform.machine() == "arm64"
                )

                if is_apple_silicon:
                    # Force CPU on Apple Silicon due to MPS compatibility issues with Kokoro TTS
                    selected = "cpu"
                    print("⚠️  Apple Silicon detected: Using CPU instead of MPS")
                    print("   (Kokoro TTS has compatibility issues with MPS)")
                else:
                    selected = "mps"
            else:
                selected = "cpu"
        elif isinstance(device, torch.device):
            selected = device
        else:
            selected = str(device)

        print(f"Using device: {selected}")
        return selected

    def get_output_path(self, filename: str) -> Path:
        """Get full path for an output file.

        Args:
            filename: Name of the output file

        Returns:
            Full path to the output file
        """
        return self.output_dir / filename

    def __repr__(self) -> str:
        """String representation of config."""
        return f"TTSConfig(output_dir={self.output_dir}, device={self.device})"


def get_device_info() -> dict:
    """Get information about available compute devices.

    Returns:
        Dictionary with device availability information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_available": True,
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

    return info


def print_device_info() -> None:
    """Print information about available compute devices."""
    info = get_device_info()

    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)

    print(f"CPU: Available")

    if info["cuda_available"]:
        print(f"CUDA: Available")
        print(f"  Device: {info['cuda_device_name']}")
        print(f"  Memory: {info['cuda_memory_total']:.2f} GB")
        print(f"  Device Count: {info['cuda_device_count']}")
    else:
        print("CUDA: Not available")

    if info["mps_available"]:
        print("MPS (Apple Silicon): Available")
    else:
        print("MPS (Apple Silicon): Not available")

    print("=" * 60)

    # Recommend best device
    if info["cuda_available"]:
        print("Recommended device: CUDA (GPU acceleration)")
    elif info["mps_available"]:
        print("Recommended device: MPS (Apple Silicon acceleration)")
    else:
        print("Recommended device: CPU (no GPU acceleration available)")

    print("=" * 60)


def setup_logging(level: str = "ERROR") -> None:
    """Set up logging configuration to reduce noise from dependencies.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    import logging

    # Silence noisy libraries
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    logging.getLogger("unstructured").setLevel(logging.ERROR)
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    print(f"Logging configured (level: {level} for noisy libraries)")
