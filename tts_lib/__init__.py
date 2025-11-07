"""TTS Library - Modular text-to-speech synthesis system.

This package provides:
- Multiple TTS backends (Kokoro, Maya1, Silero)
- PDF extraction strategies
- EPUB processing
- Audio synthesis and format conversion
- Automatic dependency management
"""

__version__ = "1.0.0"

# Make key functions easily accessible
from .synthesis import synth_string, synth_pdf, synth_epub
from .setup import install_dependencies
from .init_system import initialize_system

__all__ = [
    'synth_string',
    'synth_pdf',
    'synth_epub',
    'install_dependencies',
    'initialize_system',
]
