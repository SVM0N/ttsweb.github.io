"""Output formatters for STT transcription results.

This module provides formatters for various output formats:
- Plain text (.txt)
- SRT subtitles (.srt)
- WebVTT captions (.vtt)
- JSON with timestamps (.json)
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import timedelta


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    millis = int((td.total_seconds() % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for WebVTT format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    millis = int((td.total_seconds() % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def save_as_txt(result: Dict, output_path: Path) -> None:
    """Save transcription as plain text with optional speaker labels.

    Args:
        result: Transcription result dictionary
        output_path: Path to output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Check if we have speaker information
        segments = result.get('segments', [])
        if segments and 'speaker' in segments[0]:
            # Format with speaker labels
            current_speaker = None
            for segment in segments:
                speaker = segment.get('speaker', 'UNKNOWN')
                text = segment.get('text', '').strip()

                # Add speaker label when speaker changes
                if speaker != current_speaker:
                    f.write(f"\n[{speaker}]\n")
                    current_speaker = speaker

                f.write(f"{text}\n")
        else:
            # Plain text without speakers
            f.write(result['text'])

    print(f"✓ Saved text: {output_path}")


def save_as_srt(result: Dict, output_path: Path) -> None:
    """Save transcription as SRT subtitle format with optional speaker labels.

    Args:
        result: Transcription result dictionary with segments
        output_path: Path to output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            # Subtitle number
            f.write(f"{i}\n")

            # Timestamps
            start_time = format_timestamp_srt(segment['start'])
            end_time = format_timestamp_srt(segment['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # Text with optional speaker label
            text = segment['text'].strip()
            if 'speaker' in segment:
                text = f"[{segment['speaker']}] {text}"
            f.write(f"{text}\n\n")

    print(f"✓ Saved SRT: {output_path}")


def save_as_vtt(result: Dict, output_path: Path) -> None:
    """Save transcription as WebVTT caption format with optional speaker labels.

    Args:
        result: Transcription result dictionary with segments
        output_path: Path to output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # WebVTT header
        f.write("WEBVTT\n\n")

        for i, segment in enumerate(result['segments'], 1):
            # Optional cue identifier
            f.write(f"{i}\n")

            # Timestamps
            start_time = format_timestamp_vtt(segment['start'])
            end_time = format_timestamp_vtt(segment['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # Text with optional speaker label
            text = segment['text'].strip()
            if 'speaker' in segment:
                text = f"<v {segment['speaker']}>{text}"
            f.write(f"{text}\n\n")

    print(f"✓ Saved VTT: {output_path}")


def save_as_json(result: Dict, output_path: Path) -> None:
    """Save transcription as JSON with full details.

    Args:
        result: Transcription result dictionary
        output_path: Path to output file
    """
    # Pretty print JSON with indentation
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved JSON: {output_path}")


def save_transcription(
    result: Dict,
    base_path: Path,
    output_formats: Dict[str, bool]
) -> List[Path]:
    """Save transcription in multiple formats.

    Args:
        result: Transcription result dictionary
        base_path: Base output path (without extension)
        output_formats: Dictionary of format names to boolean enabled flags

    Returns:
        List of generated file paths
    """
    generated_files = []

    # Text format
    if output_formats.get('txt', False):
        txt_path = base_path.with_suffix('.txt')
        save_as_txt(result, txt_path)
        generated_files.append(txt_path)

    # SRT format
    if output_formats.get('srt', False):
        srt_path = base_path.with_suffix('.srt')
        save_as_srt(result, srt_path)
        generated_files.append(srt_path)

    # VTT format
    if output_formats.get('vtt', False):
        vtt_path = base_path.with_suffix('.vtt')
        save_as_vtt(result, vtt_path)
        generated_files.append(vtt_path)

    # JSON format
    if output_formats.get('json', False):
        json_path = base_path.with_suffix('.json')
        save_as_json(result, json_path)
        generated_files.append(json_path)

    return generated_files
