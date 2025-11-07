# Files Directory

This directory contains input PDFs and generated audio outputs.

## Structure

- **Input PDFs**: Place your PDF files here
- **Generated Outputs**: Audio files (`.mp3` or `.wav`) and manifests (`.json`) are saved here

## Usage

### Adding PDFs

1. Copy your PDF files to this directory
2. Update the `PDF_PATH` variable in the notebook to point to your PDF:
   ```python
   PDF_PATH = "files/your_document.pdf"
   ```

### Generated Files

When you run the TTS synthesis, the following files will be created here:

- `{filename}_tts.mp3` - Audio file in MP3 format
- `{filename}_tts_manifest.json` - Timeline manifest with sentence-level timing

### Example

If you process `Case1Writeup.pdf`, you'll get:
- `Case1Writeup_tts.mp3`
- `Case1Writeup_tts_manifest.json`

## Default Configuration

The notebook is configured to:
- Read PDFs from: `files/`
- Save outputs to: `files/`

You can change this by modifying `OUTPUT_DIR` in the notebook configuration cell.
