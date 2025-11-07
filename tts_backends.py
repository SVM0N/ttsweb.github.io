"""TTS backend implementations for different models.

This module provides TTS model backends:
- KokoroBackend: Kokoro TTS (v0.9+ and v1.0)
- SileroBackend: Silero v5 Russian TTS
"""

import io
import numpy as np
import soundfile as sf
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Optional
from functools import lru_cache

from tts_utils import split_sentences_keep_delim


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    def __init__(self, device: Union[str, torch.device] = "auto"):
        """Initialize the TTS backend.

        Args:
            device: Device to use ("auto", "cuda", "cpu", or torch.device)
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device) if isinstance(device, torch.device) else device

    @abstractmethod
    def synthesize_sentence(self, text: str, **kwargs) -> np.ndarray:
        """Synthesize a single sentence.

        Args:
            text: Text to synthesize
            **kwargs: Model-specific parameters (voice, speaker, speed, etc.)

        Returns:
            Audio as numpy array (float32)
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of this TTS backend.

        Returns:
            Sample rate in Hz
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this TTS backend.

        Returns:
            Backend name
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Get list of available voices/speakers.

        Returns:
            List of voice names
        """
        pass

    @abstractmethod
    def get_default_voice(self) -> str:
        """Get the default voice.

        Returns:
            Default voice name
        """
        pass

    def synthesize_text_to_wav(
        self,
        text_or_elements: Union[str, List[Dict]],
        **kwargs
    ) -> Tuple[bytes, List[Dict]]:
        """Synthesize text to WAV with timeline information.

        Args:
            text_or_elements: Either a string or list of text elements with metadata
            **kwargs: Model-specific parameters

        Returns:
            Tuple of (wav_bytes, timeline)
            - wav_bytes: WAV audio data
            - timeline: List of sentence timing/location data
        """
        sr = self.get_sample_rate()

        # Convert string to elements format if needed
        if isinstance(text_or_elements, str):
            elements = [{
                "text": text_or_elements,
                "metadata": {"page_number": 1, "points": None}
            }]
        else:
            elements = text_or_elements

        pcm_all = []
        timeline = []
        t = 0.0
        sentence_index = 0

        print(f"Synthesizing {len(elements)} text elements...")

        for element in elements:
            element_text = element.get("text", "")
            element_meta = element.get("metadata", {})

            sentences = split_sentences_keep_delim(element_text)

            for sent in sentences:
                if not sent:
                    continue

                pcm = self.synthesize_sentence(sent, **kwargs)
                dur = pcm.shape[0] / sr

                timeline.append({
                    "i": sentence_index,
                    "start": round(t, 3),
                    "end": round(t + dur, 3),
                    "text": sent.strip(),
                    "location": element_meta
                })

                pcm_all.append(pcm)
                t += dur
                sentence_index += 1

        # Concatenate all audio
        pcm_cat = np.concatenate(pcm_all, axis=0) if pcm_all else np.zeros((sr // 10,), dtype=np.float32)

        # Convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, pcm_cat, sr, format='WAV')
        buf.seek(0)

        return buf.read(), timeline


class KokoroBackend(TTSBackend):
    """TTS backend for Kokoro models.

    Supports Kokoro v0.9+ and v1.0 with multiple voices and languages.
    """

    def __init__(self, device: Union[str, torch.device] = "auto", version: str = "0.9"):
        """Initialize Kokoro backend.

        Args:
            device: Device to use
            version: Kokoro version ("0.9" or "1.0")
        """
        super().__init__(device)
        self.version = version
        self._pipeline_cache = {}

        print(f"Initializing Kokoro {version} backend on {self.device}...")

    @lru_cache(maxsize=4)
    def _get_pipeline(self, lang_code: str = 'a'):
        """Get or create a Kokoro pipeline for the given language.

        Args:
            lang_code: Language code (default: 'a' for auto)

        Returns:
            Kokoro pipeline instance
        """
        from kokoro import KPipeline
        return KPipeline(lang_code=lang_code, device=self.device)

    def synthesize_sentence(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = 'a',
        **kwargs
    ) -> np.ndarray:
        """Synthesize a sentence using Kokoro.

        Args:
            text: Text to synthesize
            voice: Voice to use (default: "af_heart")
            speed: Speech speed multiplier (default: 1.0)
            lang_code: Language code (default: 'a')

        Returns:
            Audio as numpy array
        """
        pipe = self._get_pipeline(lang_code)
        subchunks = []

        for _, _, audio in pipe(text, voice=voice, speed=speed, split_pattern=None):
            subchunks.append(audio)

        if not subchunks:
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(subchunks, axis=0)

    def get_sample_rate(self) -> int:
        return 24000

    def get_name(self) -> str:
        return f"Kokoro v{self.version}"

    def get_available_voices(self) -> List[str]:
        """Get available Kokoro voices.

        Note: This is a subset. Kokoro v1.0 has 54 voices across 8 languages.
        """
        if self.version == "1.0":
            return [
                # English
                "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky",
                "am_adam", "am_michael",
                "bf_emma", "bf_isabella",
                "bm_george", "bm_lewis",
                # Other languages available but not listed here for brevity
            ]
        else:
            return [
                "af_heart", "af_bella", "af_sarah", "af_nicole",
                "am_adam", "am_michael",
                "bf_emma", "bf_isabella",
                "bm_george", "bm_lewis",
            ]

    def get_default_voice(self) -> str:
        return "af_heart"


class Maya1Backend(TTSBackend):
    """TTS backend for Maya1 - Expressive multilingual TTS.

    Features:
    - 20+ emotions (laugh, cry, whisper, angry, sigh, gasp, etc.)
    - Natural language voice descriptions
    - Real-time streaming with SNAC neural codec
    - 3B parameters, optimized for single GPU
    """

    def __init__(self, device: Union[str, torch.device] = "auto"):
        """Initialize Maya1 backend."""
        super().__init__(device)
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self._load_model()

        # Maya1 token constants
        self.CODE_START_TOKEN_ID = 128257
        self.CODE_END_TOKEN_ID = 128258
        self.CODE_TOKEN_OFFSET = 128266
        self.SNAC_MIN_ID = 128266
        self.SNAC_MAX_ID = 156937
        self.SNAC_TOKENS_PER_FRAME = 7
        self.SOH_ID = 128259
        self.EOH_ID = 128260
        self.SOA_ID = 128261
        self.BOS_ID = 128000
        self.TEXT_EOT_ID = 128009

    def _load_model(self):
        """Load the Maya1 model and SNAC decoder."""
        print("Loading Maya1 model (3B parameters)...")
        print("  This requires ~16GB VRAM and may take a few minutes...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from snac import SNAC

            # Load Maya1 model
            self.model = AutoModelForCausalLM.from_pretrained(
                "maya-research/maya1",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "maya-research/maya1",
                trust_remote_code=True
            )

            # Load SNAC decoder
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            if torch.cuda.is_available():
                self.snac_model = self.snac_model.to("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Maya1 requires CUDA, will fallback to CPU if MPS
                print("  ⚠️  Warning: Maya1 requires CUDA. MPS not supported, using CPU.")
                self.snac_model = self.snac_model.to("cpu")

            print("✓ Maya1 model loaded successfully")
            print("  Sample rate: 24kHz")
            print("  Emotions: laugh, cry, whisper, angry, sigh, gasp, and more")

        except Exception as e:
            print(f"✗ Failed to load Maya1 model: {e}")
            self.model = None
            self.tokenizer = None
            self.snac_model = None

    def _build_prompt(self, description: str, text: str) -> str:
        """Build Maya1 prompt with voice description and text."""
        soh_token = self.tokenizer.decode([self.SOH_ID])
        eoh_token = self.tokenizer.decode([self.EOH_ID])
        soa_token = self.tokenizer.decode([self.SOA_ID])
        sos_token = self.tokenizer.decode([self.CODE_START_TOKEN_ID])
        eot_token = self.tokenizer.decode([self.TEXT_EOT_ID])
        bos_token = self.tokenizer.bos_token

        formatted_text = f'<description="{description}"> {text}'
        prompt = (
            soh_token + bos_token + formatted_text + eot_token +
            eoh_token + soa_token + sos_token
        )
        return prompt

    def _extract_snac_codes(self, token_ids: list) -> list:
        """Extract SNAC codes from generated token IDs."""
        try:
            eos_idx = token_ids.index(self.CODE_END_TOKEN_ID)
        except ValueError:
            eos_idx = len(token_ids)

        snac_codes = [
            token_id for token_id in token_ids[:eos_idx]
            if self.SNAC_MIN_ID <= token_id <= self.SNAC_MAX_ID
        ]
        return snac_codes

    def _unpack_snac_from_7(self, snac_tokens: list) -> list:
        """Unpack SNAC tokens from 7-token format to 3 hierarchical levels."""
        if snac_tokens and snac_tokens[-1] == self.CODE_END_TOKEN_ID:
            snac_tokens = snac_tokens[:-1]

        frames = len(snac_tokens) // self.SNAC_TOKENS_PER_FRAME
        snac_tokens = snac_tokens[:frames * self.SNAC_TOKENS_PER_FRAME]

        l1, l2, l3 = [], [], []

        for i in range(frames):
            slots = snac_tokens[i*7:(i+1)*7]
            l1.append((slots[0] - self.CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - self.CODE_TOKEN_OFFSET) % 4096,
            ])
            l3.extend([
                (slots[2] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - self.CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - self.CODE_TOKEN_OFFSET) % 4096,
            ])

        return [l1, l2, l3]

    def synthesize_sentence(
        self,
        text: str,
        voice: str = "Realistic male voice in the 30s with American accent",
        **kwargs
    ) -> np.ndarray:
        """Synthesize a sentence using Maya1.

        Args:
            text: Text to synthesize (can include emotion tags like <laugh>)
            voice: Natural language voice description

        Returns:
            Audio as numpy array
        """
        if self.model is None or self.tokenizer is None or self.snac_model is None:
            print("Warning: Maya1 model not loaded, returning silence")
            return np.zeros((self.get_sample_rate() // 10,), dtype=np.float32)

        try:
            # Build prompt
            prompt = self._build_prompt(voice, text)
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move to device
            device = self.model.device if hasattr(self.model, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate tokens
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    min_new_tokens=28,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=self.CODE_END_TOKEN_ID,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Extract SNAC codes
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
            snac_tokens = self._extract_snac_codes(generated_ids)
            levels = self._unpack_snac_from_7(snac_tokens)

            # Convert to audio using SNAC decoder
            snac_device = next(self.snac_model.parameters()).device
            codes_tensor = [
                torch.tensor(level, dtype=torch.long, device=snac_device).unsqueeze(0)
                for level in levels
            ]

            with torch.inference_mode():
                z_q = self.snac_model.quantizer.from_codes(codes_tensor)
                audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()

            # Trim warmup samples
            audio = audio[2048:]

            return audio.astype(np.float32)

        except Exception as e:
            print(f"Error synthesizing sentence: {e}")
            return np.zeros((self.get_sample_rate() // 10,), dtype=np.float32)

    def get_sample_rate(self) -> int:
        return 24000

    def get_name(self) -> str:
        return "Maya1 (Expressive)"

    def get_available_voices(self) -> List[str]:
        """Get example voice descriptions.

        Note: Maya1 uses natural language descriptions, not preset voices.
        """
        return [
            "Realistic male voice in the 30s with American accent",
            "Warm female voice in the 40s, conversational",
            "Young energetic male voice, British accent",
            "Calm female voice, professional tone",
            "Deep male voice, authoritative",
        ]

    def get_default_voice(self) -> str:
        return "Realistic male voice in the 30s with American accent"


class SileroBackend(TTSBackend):
    """TTS backend for Silero v5 Russian TTS.

    Specialized for Russian language synthesis with multiple speakers.
    """

    def __init__(self, device: Union[str, torch.device] = "auto"):
        """Initialize Silero backend."""
        super().__init__(device)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Silero v5 Russian model."""
        print("Loading Silero v5 Russian TTS model...")
        try:
            self.model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v5_ru'
            )

            # Move to device
            if isinstance(self.device, str):
                device_obj = torch.device(self.device)
            else:
                device_obj = self.device

            self.model.to(device_obj)
            print(f"✓ Silero v5 Russian model loaded successfully on {self.device}")
            print(f"Example text: {example_text[:50]}...")

        except Exception as e:
            print(f"✗ Failed to load Silero model: {e}")
            self.model = None

    def synthesize_sentence(
        self,
        text: str,
        speaker: str = "xenia",
        **kwargs
    ) -> np.ndarray:
        """Synthesize a sentence using Silero v5.

        Args:
            text: Text to synthesize (should be Russian)
            speaker: Speaker voice (default: "xenia")

        Returns:
            Audio as numpy array
        """
        if self.model is None:
            print("Warning: Silero model not loaded, returning silence")
            return np.zeros((self.get_sample_rate() // 10,), dtype=np.float32)

        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self.get_sample_rate()
            )

            # Convert tensor to numpy if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            return audio.astype(np.float32)

        except Exception as e:
            print(f"Error synthesizing sentence: {e}")
            return np.zeros((self.get_sample_rate() // 10,), dtype=np.float32)

    def get_sample_rate(self) -> int:
        return 48000

    def get_name(self) -> str:
        return "Silero v5 Russian"

    def get_available_voices(self) -> List[str]:
        """Get available Silero speakers."""
        return ['xenia', 'eugene', 'baya', 'kseniya', 'aleksandr', 'irina']

    def get_default_voice(self) -> str:
        return "xenia"


def get_available_backends() -> Dict[str, type]:
    """Get a dictionary of all available TTS backends.

    Returns:
        Dictionary mapping backend names to backend classes
    """
    return {
        "kokoro_0.9": lambda device: KokoroBackend(device=device, version="0.9"),
        "kokoro_1.0": lambda device: KokoroBackend(device=device, version="1.0"),
        "maya1": Maya1Backend,
        "silero_v5": SileroBackend,
    }


def create_backend(backend_name: str, device: Union[str, torch.device] = "auto") -> TTSBackend:
    """Create a TTS backend by name.

    Args:
        backend_name: Name of the backend ("kokoro_0.9", "kokoro_1.0", "maya1", "silero_v5")
        device: Device to use

    Returns:
        TTS backend instance

    Raises:
        ValueError: If backend_name is not recognized
    """
    backends = get_available_backends()

    if backend_name not in backends:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available backends: {list(backends.keys())}"
        )

    return backends[backend_name](device)
