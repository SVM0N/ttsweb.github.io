"""System initialization for TTS notebook.

This module handles:
- Importing all required modules
- Setting up logging
- Initializing TTS backend
- Initializing PDF extractor
"""

from config import TTSConfig, print_device_info, setup_logging
from tts_backends import create_backend
from synthesis import synth_string, synth_pdf, synth_epub


def import_modules(enable_pdf_input, pdf_extractor):
    """Import required modules and set up logging.

    Args:
        enable_pdf_input: Whether PDF input is enabled
        pdf_extractor: PDF extractor name (or None)

    Returns:
        pdf_extractors module if PDF input is enabled, else None
    """
    print("\n📥 Importing modules...")

    # Import PDF extractor if needed
    pdf_extractors = None
    if enable_pdf_input and pdf_extractor:
        from pdf_extractors import get_available_extractors
        pdf_extractors = get_available_extractors

    # Set up logging to reduce noise
    setup_logging()

    print("✓ Modules imported successfully")

    return pdf_extractors


def initialize_system(
    tts_model,
    output_dir,
    device,
    pdf_extractor_name=None,
    enable_pdf_input=False
):
    """Initialize the TTS system.

    Args:
        tts_model: TTS model name
        output_dir: Output directory path
        device: Device to use ("auto", "cuda", "cpu", "mps")
        pdf_extractor_name: PDF extractor name (or None)
        enable_pdf_input: Whether PDF input is enabled

    Returns:
        Tuple of (tts_backend, config, pdf_extractor)
    """
    # Print available devices
    print_device_info()

    # Create configuration
    config = TTSConfig(output_dir=output_dir, device=device)
    print(f"\n{config}")

    # Load TTS backend
    print(f"\n📥 Loading TTS backend: {tts_model}...")
    tts = create_backend(tts_model, device=config.device)
    print(f"✓ TTS backend loaded: {tts.get_name()}")
    print(f"  Available voices: {tts.get_available_voices()[:5]}...")  # Show first 5
    print(f"  Default voice: {tts.get_default_voice()}")
    print(f"  Sample rate: {tts.get_sample_rate()} Hz")

    # Load PDF extractor if needed
    pdf_extractor = None
    if enable_pdf_input and pdf_extractor_name:
        print(f"\n📥 Loading PDF extractor: {pdf_extractor_name}...")
        from pdf_extractors import get_available_extractors
        extractors = get_available_extractors()
        pdf_extractor = extractors[pdf_extractor_name]
        print(f"✓ PDF extractor loaded: {pdf_extractor.get_name()}")
        print(f"  Description: {pdf_extractor.get_description()}")
    else:
        print("\n⚠️  PDF extraction disabled (PDF_EXTRACTOR not set)")

    print("\n✓ System initialized and ready!")

    return tts, config, pdf_extractor
