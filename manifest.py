"""Manifest generation for TTS timeline data.

This module provides utilities for creating and managing TTS manifest files
that contain sentence-level timing information and coordinate data.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


def create_manifest(audio_filename: str, timeline: List[Dict]) -> Dict:
    """Create a manifest dictionary.

    Args:
        audio_filename: Name of the audio file
        timeline: List of sentence timing/location dictionaries

    Returns:
        Manifest dictionary with audioUrl and sentences
    """
    return {
        "audioUrl": audio_filename,
        "sentences": timeline
    }


def save_manifest(manifest: Dict, output_path: str) -> None:
    """Save a manifest to a JSON file.

    Args:
        manifest: Manifest dictionary
        output_path: Path to save the manifest file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_manifest(manifest_path: str) -> Dict:
    """Load a manifest from a JSON file.

    Args:
        manifest_path: Path to the manifest file

    Returns:
        Manifest dictionary
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest: Dict) -> bool:
    """Validate a manifest structure.

    Args:
        manifest: Manifest dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    if "audioUrl" not in manifest:
        return False

    if "sentences" not in manifest:
        return False

    if not isinstance(manifest["sentences"], list):
        return False

    # Check each sentence entry
    for sentence in manifest["sentences"]:
        required_fields = ["i", "start", "end", "text", "location"]
        if not all(field in sentence for field in required_fields):
            return False

        # Check location structure
        location = sentence["location"]
        if not isinstance(location, dict):
            return False

        if "page_number" not in location:
            return False

    return True


def merge_manifests(manifests: List[Dict], audio_filename: str) -> Dict:
    """Merge multiple manifests into a single manifest.

    Useful for combining chapter manifests or multi-part synthesis.

    Args:
        manifests: List of manifest dictionaries
        audio_filename: Name for the combined audio file

    Returns:
        Merged manifest dictionary
    """
    merged_sentences = []
    sentence_index = 0
    time_offset = 0.0

    for manifest in manifests:
        for sentence in manifest.get("sentences", []):
            merged_sentence = {
                "i": sentence_index,
                "start": round(sentence["start"] + time_offset, 3),
                "end": round(sentence["end"] + time_offset, 3),
                "text": sentence["text"],
                "location": sentence["location"]
            }
            merged_sentences.append(merged_sentence)
            sentence_index += 1

        # Update time offset for next manifest
        if manifest.get("sentences"):
            last_sentence = manifest["sentences"][-1]
            time_offset = last_sentence["end"] + time_offset

    return create_manifest(audio_filename, merged_sentences)


def get_manifest_stats(manifest: Dict) -> Dict:
    """Get statistics about a manifest.

    Args:
        manifest: Manifest dictionary

    Returns:
        Dictionary with statistics:
        - total_sentences: Number of sentences
        - total_duration: Total audio duration in seconds
        - pages: Set of unique page numbers
        - avg_sentence_duration: Average sentence duration
    """
    sentences = manifest.get("sentences", [])

    if not sentences:
        return {
            "total_sentences": 0,
            "total_duration": 0.0,
            "pages": set(),
            "avg_sentence_duration": 0.0
        }

    total_duration = sentences[-1]["end"] if sentences else 0.0
    pages = set()

    for sentence in sentences:
        location = sentence.get("location", {})
        page_num = location.get("page_number")
        if page_num:
            pages.add(page_num)

    avg_duration = total_duration / len(sentences) if sentences else 0.0

    return {
        "total_sentences": len(sentences),
        "total_duration": round(total_duration, 3),
        "pages": sorted(pages),
        "avg_sentence_duration": round(avg_duration, 3)
    }


def print_manifest_summary(manifest: Dict) -> None:
    """Print a human-readable summary of a manifest.

    Args:
        manifest: Manifest dictionary
    """
    stats = get_manifest_stats(manifest)

    print("=" * 60)
    print("MANIFEST SUMMARY")
    print("=" * 60)
    print(f"Audio file: {manifest.get('audioUrl', 'N/A')}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total duration: {stats['total_duration']:.2f} seconds")
    print(f"Average sentence duration: {stats['avg_sentence_duration']:.2f} seconds")
    print(f"Pages: {stats['pages']}")
    print("=" * 60)
