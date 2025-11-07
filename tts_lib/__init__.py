"""TTS Library - Modular text-to-speech synthesis system.

This package provides:
- Multiple TTS backends (Kokoro, Maya1, Silero)
- PDF extraction strategies
- EPUB processing
- Audio synthesis and format conversion
- Automatic dependency management
"""

__version__ = "1.0.0"

# Don't import here to avoid circular imports
# Users should import directly from submodules:
# from tts_lib.setup import install_dependencies
# from tts_lib.init_system import initialize_system
# from tts_lib.synthesis import synth_string, synth_pdf, synth_epub
