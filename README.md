# TTS Web - PDF to Synchronized Audio

Convert PDFs and EPUBs into audiobooks with synchronized text highlighting using state-of-the-art text-to-speech models.

<details>
<summary><strong>📑 Table of Contents</strong></summary>

- [What This Does](#-what-this-does)
- [Getting Started](#-getting-started)
  - [Quick Start with Unified Notebook](#quick-start-with-unified-notebook-)
  - [Traditional Setup](#traditional-setup-legacy-notebooks)
  - [Google Colab Setup](#google-colab-setup)
- [Modular Architecture](#-modular-architecture)
  - [Core Modules](#core-modules)
  - [Benefits](#benefits-of-modular-design)
  - [Programmatic Usage](#using-the-modules-programmatically)
- [Available Models](#-available-models--when-to-use-each)
- [Output Files](#-output-files)
- [Web Player](#-using-the-web-player)
- [Customization](#-customization)
  - [Voice Selection](#voice-selection-kokoro-models)
  - [Voice Cloning](#voice-cloning-f5-tts-mlx)
  - [Output Format](#output-format)
  - [Speech Speed](#speech-speed)
- [Managing Caches](#-managing-model-caches)
- [Decision Guide](#-quick-decision-guide)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Credits](#-credits)
- [Legacy Notebooks](#legacy-notebooks)

</details>

---

## 🎯 What This Does
This project provides Jupyter notebooks that:
1. Extract text from PDFs/EPUBs with precise coordinate tracking
2. Generate high-quality speech audio using AI TTS models
3. Create timeline manifests for synchronized text highlighting
4. Output files ready to upload to the web player at **https://svm0n.github.io/ttsweb.github.io/**

## 🚀 Getting Started

### Quick Start with Unified Notebook ⭐ NEW

**The easiest way to use this project is with the new unified notebook:**

1. Clone this repository:
   ```bash
   git clone https://github.com/SVM0N/ttsweb.github.io.git
   cd ttsweb.github.io
   ```

2. Open the unified notebook:
   ```bash
   jupyter notebook TTS.ipynb
   ```

3. Follow the notebook instructions to:
   - Create an isolated conda environment (optional but recommended)
   - Choose your TTS model (Kokoro v0.9, Kokoro v1.0, Silero v5)
   - Choose your PDF extractor (Unstructured, PyMuPDF, Vision, Nougat)
   - Run synthesis on text, PDFs, or EPUBs

**The unified notebook (`TTS.ipynb`) combines all models and extractors in one place!**

**Benefits of the unified notebook:**
- ✨ **Smart dependency installation**: Only installs packages you actually need
- 🎯 **Easy configuration**: Choose models/extractors in one cell at the top
- 💾 **Saves storage**: No need to install everything upfront
- 🔄 **Easy switching**: Change configuration and re-run without reinstalling

### Traditional Setup (Legacy Notebooks)

**Legacy notebooks have been moved to the `archived/` folder.** You can still use them if you prefer the old standalone approach, but the unified notebook is recommended for new users.

**Prerequisites:**
- Python 3.10+
- conda (recommended) or pip
- ffmpeg (for MP3 conversion)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
  - Windows: Download from https://ffmpeg.org/

**Steps:**
1. Clone this repository:
   ```bash
   git clone https://github.com/SVM0N/ttsweb.github.io.git
   cd ttsweb.github.io
   ```

2. Choose a notebook (see "Which Model to Use" below)

3. Open the notebook in Jupyter:
   ```bash
   jupyter notebook TTS_Kokoro_Local.ipynb
   ```

4. Follow the notebook instructions to create an isolated conda environment (recommended)

5. Run all cells and provide your PDF/EPUB file path when prompted

### Google Colab Setup

1. Visit this repository on GitHub: https://github.com/SVM0N/ttsweb.github.io

2. Click on the notebook you want to use (e.g., `TTS_Kokoro_Local.ipynb`)

3. Click "Open in Colab" button (or manually upload to Colab)

4. Upload your PDF/EPUB file to Colab using the file upload interface:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

5. Run all cells

6. Download the generated audio and manifest files from Colab

## 🏗️ Modular Architecture

This project now features a **modular Python architecture** that makes it easy to:
- Switch between different TTS models without code duplication
- Choose PDF extraction strategies based on your needs
- Extend functionality with custom backends

### Core Modules

**`TTS.ipynb`** - Unified notebook interface
- Single notebook for all TTS models and PDF extractors
- Interactive model and extractor selection
- No code duplication across notebooks

**`tts_backends.py`** - TTS model backends
- `KokoroBackend`: Kokoro TTS (v0.9 and v1.0)
- `SileroBackend`: Silero v5 Russian TTS
- Extensible for adding new models

**`pdf_extractors.py`** - PDF extraction strategies
- `UnstructuredExtractor`: Advanced layout analysis (default)
- `PyMuPDFExtractor`: Fast extraction for clean PDFs
- `VisionExtractor`: OCR for scanned PDFs (macOS only)
- `NougatExtractor`: Academic papers with equations

**`tts_utils.py`** - Common utilities
- EPUB extraction
- Sentence splitting
- WAV to MP3 conversion
- File naming utilities

**`manifest.py`** - Manifest generation
- Timeline creation with sentence-level timing
- Coordinate tracking for synchronized highlighting
- Manifest validation and statistics

**`config.py`** - Configuration management
- Device selection (CUDA/CPU/MPS)
- Output directory management
- Logging configuration

### Benefits of Modular Design

- **No Code Duplication**: Common functionality shared across all notebooks
- **Easy to Extend**: Add new TTS models or PDF extractors as plugins
- **Mix and Match**: Combine any TTS model with any PDF extractor
- **Better Testing**: Each module can be tested independently
- **Cleaner Codebase**: Easier to maintain and debug

### Using the Modules Programmatically

You can also use the modules directly in your own Python scripts:

```python
from config import TTSConfig
from tts_backends import create_backend
from pdf_extractors import get_available_extractors

# Configure
config = TTSConfig(output_dir=".", device="auto")

# Create TTS backend
tts = create_backend("kokoro_1.0", device=config.device)

# Get PDF extractor
extractors = get_available_extractors()
pdf_extractor = extractors["pymupdf"]

# Extract and synthesize
pdf_bytes = open("document.pdf", "rb")
elements = pdf_extractor.extract(pdf_bytes)
wav_bytes, timeline = tts.synthesize_text_to_wav(elements, voice="af_heart")
```

## 📚 Available Models & When to Use Each

### **NEW: TTS.ipynb** ⭐ **UNIFIED NOTEBOOK**

**When to use:**
- You want a single notebook that supports all models
- You want to easily switch between TTS models
- You want to try different PDF extraction strategies
- You prefer a clean, modular interface

**Supported TTS Models:**
- Kokoro v0.9.4+ (10 voices, English-focused)
- Kokoro v1.0 (54 voices, 8 languages)
- Silero v5 (Russian, 6 speakers)

**Supported PDF Extractors:**
- Unstructured (advanced layout analysis)
- PyMuPDF (fast, lightweight)
- Apple Vision (OCR, macOS only)
- Nougat (academic papers)

**Pros:**
- All-in-one solution
- No code duplication
- Easy to switch between models
- Modular and extensible

**Cons:**
- Requires all module files (tts_backends.py, pdf_extractors.py, etc.)

---

## 🎬 Output Files

Each notebook generates two files:

### 1. Audio File
- **Format:** MP3 or WAV (configurable)
- **Sample Rate:** 24kHz (F5-MLX) or 24kHz (Kokoro)
- **Naming:** `{filename}_tts.mp3` or `{filename}_tts.wav`

### 2. Manifest File
- **Format:** JSON
- **Naming:** `{filename}_tts_manifest.json`
- **Contains:**
  - Sentence-level timestamps
  - Text content for each segment
  - Coordinate data for highlighting (page number, bounding boxes)

**Example manifest structure:**
```json
{
  "audioUrl": "document_tts.mp3",
  "sentences": [
    {
      "i": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "This is the first sentence.",
      "location": {
        "page_number": 1,
        "points": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
      }
    }
  ]
}
```

## 🌐 Using the Web Player

1. **Generate your files** using any notebook above

2. **Upload to the web player** at: **https://svm0n.github.io/ttsweb.github.io/**

3. **Upload both files:**
   - Your PDF file
   - The audio file (MP3/WAV)
   - The manifest JSON file

4. **Play and enjoy** synchronized audio with text highlighting!

The web player features:
- PDF rendering with synchronized highlighting
- Audio playback controls (play/pause, seek, speed control)
- Click on text to jump to that audio position
- Dark mode support
- Mobile-friendly responsive design

## 🛠️ Customization

### Voice Selection (Kokoro models)

**Kokoro v0.9.x (TTS_Kokoro_Local.ipynb):**
- Available voices: `af_heart` , `af_bella`, `af_sarah`, `am_adam`, `am_michael`, and more

**Kokoro v1.0 (TTS_Kokoro_v.1.0_Local.ipynb):**
- 54 voices across 8 languages (see full list in section 2 above)
- US, British, French, Japanese, Korean, Chinese voices available

```python
VOICE = "af_heart"  # Change to any available voice
```

### Voice Cloning (F5-TTS-MLX)
Provide reference audio for zero-shot voice cloning:

```python
REF_AUDIO = "reference.wav"  # 5-10s mono WAV at 24kHz
REF_TEXT = "This is what the speaker says in the reference audio."
```

Convert your audio:
```bash
ffmpeg -i input.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 reference.wav
```

### Output Format
```python
FORMAT = "mp3"  # or "wav"
```

### Speech Speed
```python
SPEED = 1.0  # 0.5 = half speed, 2.0 = double speed
```

## 🗑️ Managing Model Caches

TTS models are cached locally to improve performance. Each notebook includes a cache management section at the end where you can:

- **View cache locations and sizes** for:
  - HuggingFace models (`~/.cache/huggingface/`)
  - PyTorch models (`~/.cache/torch/`)
  - Pip packages
  - Model-specific caches

- **Delete cached models** to free up storage:
  - Individual model deletion
  - Bulk cache cleanup
  - Environment-specific cleanup

**Typical cache sizes:**
- Kokoro models: ~500MB - 1GB
- F5-TTS-MLX models: ~300MB - 500MB
- Detectron2 models: ~200MB - 400MB
- Nougat models: ~1GB - 2GB

Each local notebook includes an optional cleanup section at the end to help manage these caches.

## 📋 Quick Decision Guide

**I want the easiest, most flexible option:**
→ Use **TTS.ipynb** ⭐ (Unified notebook - recommended for everyone)

**I need Russian language TTS:**
→ Use **TTS.ipynb** with Silero v5 backend

**I have Apple Silicon (M1/M2/M3/M4):**
→ Use **TTS_F5_MLX.ipynb** (archived/TTS_F5_MLX.ipynb)

**I need maximum speed and my PDF has text:**
→ Use **TTS.ipynb** with PyMuPDF extractor

**I have a scanned PDF (no text layer):**
→ Use **TTS.ipynb** with Vision/Nougat extractor

**I have an academic paper with equations:**
→ Use **TTS.ipynb** with Nougat extractor

**I prefer the old standalone notebooks:**
→ Check the `archived/` folder for legacy notebooks

## 🔧 Troubleshooting

### "NotImplementedError: aten::angle not implemented for MPS"
- Add this before imports: `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'`
- Restart your Jupyter kernel

### "Highlights are off by 1-2 lines"
- This has been fixed in the latest version
- Make sure you're using the updated notebooks

### "Out of memory"
- Use **TTS_Kokoro_PyMuPDF.ipynb** (lowest memory usage)
- Or process shorter documents
- Or add more RAM/swap space

### "PDF extraction failed"
- If using PyMuPDF: Your PDF might be scanned → Use Vision or Nougat
- If using Vision: Check macOS version compatibility
- If using Nougat: Ensure GPU is available and CUDA installed

## 📄 License

   This project is licensed for non-commercial use only.
   For commercial licensing, please contact SVM0N on GitHub.

## 🙏 Credits

This project uses:
- [Kokoro TTS](https://github.com/remsky/Kokoro-82M) - High-quality text-to-speech
- [F5-TTS-MLX](https://github.com/lucasnewman/f5-tts-mlx) - Apple Silicon optimized TTS
- [Unstructured.io](https://github.com/Unstructured-IO/unstructured) - Document parsing
- [Detectron2](https://github.com/facebookresearch/detectron2) - Layout detection
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - Fast PDF processing
- [Nougat](https://github.com/facebookresearch/nougat) - Academic document OCR
- [PDF.js](https://mozilla.github.io/pdf.js/) - Web PDF rendering

---

<details>
<summary>

### **Legacy Notebooks**


</summary>

The following notebooks have been moved to the `archived/` folder and are still available for backwards compatibility:

### **1. TTS_Kokoro_Local.ipynb** ⭐ **RECOMMENDED DEFAULT**

**When to use:**
- General purpose, works on most machines
- Best balance of quality, speed, and coordinate accuracy
- PDF extraction using ML-based layout analysis
- Stable version with Kokoro v0.9.4+

**Machine requirements:**
- RAM: 8GB minimum, 16GB recommended
- GPU: Optional (CUDA) but works fine on CPU
- Storage: ~5GB for dependencies

**Pros:**
- Excellent text extraction for complex layouts
- Accurate bounding box coordinates
- Multiple voice options (10 voices)
- Reliable and well-tested

**Cons:**
- Slower PDF processing than PyMuPDF
- Larger dependency footprint

---

### **2. TTS_Kokoro_v.1.0_Local.ipynb** 🆕 **LATEST KOKORO**

**When to use:**
- Want the latest Kokoro v1.0 features
- Need access to 54 voices across 8 languages
- Want voice blending capabilities
- Multi-language support (French, Japanese, Korean, Chinese, etc.)

**Machine requirements:**
- RAM: 8GB minimum, 16GB recommended
- GPU: Optional (CUDA) but works fine on CPU
- Storage: ~5GB for dependencies

**Pros:**
- 54 voices (vs 10 in v0.9.x)
- 8 languages (vs 1 in v0.9.x)
- Voice blending for custom voices
- Same API as v0.9.x (backward compatible)
- Trained on hundreds of hours of audio

**Cons:**
- Newer, less battle-tested than v0.9.x
- Larger model downloads

**Available voices:**
- US Female (11): af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
- US Male (8): am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck
- British Female (4): bf_alice, bf_emma, bf_isabella, bf_lily
- British Male (4): bm_daniel, bm_fable, bm_george, bm_lewis
- Plus voices for French, Japanese, Korean, Chinese, and more

---

### **3. TTS_F5_MLX.ipynb** 🍎 **BEST FOR APPLE SILICON**

**When to use:**
- You have Apple Silicon (M1/M2/M3/M4)
- Want maximum performance on Mac
- Need voice cloning capabilities

**Machine requirements:**
- Apple Silicon Mac (M1/M2/M3/M4)
- RAM: 8GB minimum, 16GB recommended
- Storage: ~5GB for dependencies

**Pros:**
- Optimized for Apple's MLX framework
- Excellent multicore utilization
- Zero-shot voice cloning support
- ~4 seconds per sentence on M3/M4

**Cons:**
- Apple Silicon only (CPU fallback available but slow)
- Requires reference audio for voice cloning

**Voice cloning:** Provide a 5-10 second mono WAV file (24kHz) and its transcription to clone any voice.

---

### **4. TTS_Kokoro_PyMuPDF.ipynb** ⚡ **FASTEST**

**When to use:**
- PDF has a clean text layer (not scanned)
- Need fastest possible processing
- Have limited RAM/storage

**Machine requirements:**
- RAM: 4GB minimum
- CPU: Any modern CPU
- Storage: ~2GB for dependencies

**Pros:**
- Extremely fast PDF text extraction
- Minimal dependencies
- Low resource usage
- Very accurate coordinates

**Cons:**
- Only works for PDFs with text layers
- Fails on scanned PDFs or images
- No layout analysis

---

### **5. TTS_Kokoro_Vision.ipynb** 🔍 **FOR SCANNED PDFs**

**When to use:**
- PDF is scanned (no text layer)
- Need OCR capabilities
- macOS with Vision Framework

**Machine requirements:**
- macOS 10.15+
- RAM: 8GB minimum
- Storage: ~3GB for dependencies

**Pros:**
- Works on scanned/image-based PDFs
- Uses Apple's Vision Framework OCR
- Good for documents without text layers

**Cons:**
- macOS only
- Slower than direct text extraction
- OCR may have accuracy issues

---

### **6. TTS_Silero_v5_Local.ipynb** **FOR RUSSIAN LANGUAGE**

**When to use:**
- Need Russian language text-to-speech
- Want high-quality Russian voice synthesis
- Processing Russian documents

**Machine requirements:**
- RAM: 8GB minimum
- GPU: Optional (CUDA) but works fine on CPU
- Storage: ~3GB for dependencies

**Pros:**
- Excellent Russian pronunciation
- 6 different speakers (xenia, eugene, baya, kseniya, aleksandr, irina)
- SSML support with automated stress and homographs
- Fast synthesis

**Cons:**
- Russian language only
- Limited to 6 speakers

---

### **7. TTS_Nougat.ipynb** 📄 **FOR ACADEMIC PAPERS**

**When to use:**
- Processing academic papers with equations
- Need LaTeX/math support
- Document has complex formatting

**Machine requirements:**
- RAM: 16GB recommended
- GPU: Highly recommended (CUDA)
- Storage: ~8GB for dependencies

**Pros:**
- Excellent for academic documents
- Handles equations and math notation
- LaTeX support

**Cons:**
- Very slow without GPU
- Large model downloads
- Overkill for simple text


</details>

---

**Made with ❤️ for accessible reading**