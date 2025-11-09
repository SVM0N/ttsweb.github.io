"""Cleanup utilities for managing conda environments and model caches.

This module provides functions for:
- Deleting conda environments created during notebook sessions
- Cleaning up model and package caches to free disk space
"""

import shutil
import subprocess
from pathlib import Path


def get_dir_size(path):
    """Get the total size of a directory in bytes.

    Args:
        path: Path to directory

    Returns:
        Total size in bytes
    """
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def format_bytes(bytes_size):
    """Format bytes into human-readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def delete_cache(cache_name):
    """Delete a specific cache directory.

    Args:
        cache_name: Name of cache to delete. Options:
            - "huggingface": HuggingFace model cache
            - "torch_hub": PyTorch Hub cache
            - "torch_checkpoints": PyTorch checkpoints
            - "detectron2": Detectron2 cache
            - "kokoro": Kokoro model cache
            - "pip": Pip package cache

    Returns:
        True if deletion was successful, False otherwise
    """
    home = Path.home()

    cache_paths = {
        "huggingface": home / ".cache" / "huggingface",
        "torch_hub": home / ".cache" / "torch" / "hub",
        "torch_checkpoints": home / ".cache" / "torch" / "checkpoints",
        "detectron2": home / ".torch" / "fvcore_cache" / "detectron2",
        "kokoro": home / ".cache" / "kokoro",
        "pip": home / ".cache" / "pip",
    }

    if cache_name not in cache_paths:
        print(f"✗ Unknown cache: {cache_name}")
        print(f"Available caches: {', '.join(cache_paths.keys())}")
        return False

    cache_path = cache_paths[cache_name]

    if not cache_path.exists():
        print(f"⚪ Cache not found: {cache_path}")
        return False

    size_before = get_dir_size(cache_path)
    print(f"→ Deleting {cache_name} cache...")
    print(f"  Path: {cache_path}")
    print(f"  Size: {format_bytes(size_before)}")

    try:
        shutil.rmtree(cache_path)
        print(f"✓ Successfully deleted {cache_name} cache")
        print(f"  Freed: {format_bytes(size_before)}")
        return True
    except Exception as e:
        print(f"✗ Failed to delete cache: {e}")
        return False


def list_cache_sizes():
    """List all caches with their sizes and existence status.

    Returns:
        Dictionary mapping cache names to (exists, size_bytes, path)
    """
    home = Path.home()

    cache_paths = {
        "huggingface": home / ".cache" / "huggingface",
        "torch_hub": home / ".cache" / "torch" / "hub",
        "torch_checkpoints": home / ".cache" / "torch" / "checkpoints",
        "detectron2": home / ".torch" / "fvcore_cache" / "detectron2",
        "kokoro": home / ".cache" / "kokoro",
        "pip": home / ".cache" / "pip",
    }

    cache_info = {}
    for name, path in cache_paths.items():
        if path.exists():
            size = get_dir_size(path)
            cache_info[name] = (True, size, path)
        else:
            cache_info[name] = (False, 0, path)

    return cache_info


def interactive_cache_cleanup():
    """Interactive cache cleanup with user prompts."""
    print("=" * 80)
    print("CACHE INSPECTION & CLEANUP")
    print("=" * 80)
    print()

    # Show current cache status
    cache_info = list_cache_sizes()
    total_size = 0

    print("Current cache status:")
    print("-" * 80)
    cache_display = {
        "huggingface": "HuggingFace Cache",
        "torch_hub": "Torch Hub",
        "torch_checkpoints": "Torch Checkpoints",
        "detectron2": "Detectron2",
        "kokoro": "Kokoro Models",
        "pip": "Pip Cache",
    }

    for cache_name, display_name in cache_display.items():
        exists, size, path = cache_info[cache_name]
        if exists:
            size_str = format_bytes(size)
            print(f"  ✓ {display_name:25s} {size_str:>12s}  ({path})")
            total_size += size
        else:
            print(f"  ✗ {display_name:25s} {'Not found':>12s}")

    print("-" * 80)
    print(f"  TOTAL:                      {format_bytes(total_size):>12s}")
    print()

    print("=" * 80)
    print("DELETE CACHE OPTIONS")
    print("=" * 80)
    print()
    print("Available caches to delete:")
    print("  [1] HuggingFace Cache")
    print("  [2] Torch Hub")
    print("  [3] Torch Checkpoints")
    print("  [4] Detectron2")
    print("  [5] Kokoro Models")
    print("  [6] Pip Cache")
    print("  [7] ALL caches (⚠️  WARNING: Deletes everything!)")
    print("  [0] Cancel")
    print()

    choice = input("Enter choice (0-7): ").strip()

    cache_map = {
        "1": "huggingface",
        "2": "torch_hub",
        "3": "torch_checkpoints",
        "4": "detectron2",
        "5": "kokoro",
        "6": "pip",
    }

    if choice == "0":
        print("\n✗ Deletion cancelled")
    elif choice == "7":
        confirm = input("\n⚠️  Delete ALL caches? Type 'yes' to confirm: ").strip().lower()
        if confirm == "yes":
            print("\n→ Deleting all caches...")
            for cache_name in cache_map.values():
                if delete_cache(cache_name):
                    print()
            print("=" * 80)
            print("✓ All caches deleted")
            print("=" * 80)
        else:
            print("\n✗ Deletion cancelled")
    elif choice in cache_map:
        cache_name = cache_map[choice]
        confirm = input(f"\nDelete {cache_name} cache? Type 'yes' to confirm: ").strip().lower()
        if confirm == "yes":
            print()
            delete_cache(cache_name)
        else:
            print("\n✗ Deletion cancelled")
    else:
        print("\n✗ Invalid choice")


def delete_conda_environment(environment_name, environment_created_by_notebook):
    """Delete a conda environment created during the notebook session.

    Args:
        environment_name: Name of the environment to delete
        environment_created_by_notebook: Boolean flag indicating if environment
            was created in this notebook session

    Returns:
        Tuple of (success, new_environment_created_flag, new_environment_name)
    """
    if not environment_created_by_notebook:
        print("✗ No environment was created by this notebook")
        print("You can only delete environments that were created in this session")
        return False, environment_created_by_notebook, environment_name

    print(f"Environment '{environment_name}' was created by this notebook")
    print(f"\n{'='*60}")
    print("DELETE ENVIRONMENT")
    print(f"{'='*60}")

    confirm = input(f"\nAre you sure you want to DELETE '{environment_name}'?\nType 'yes' to confirm: ").strip().lower()

    if confirm == 'yes':
        print(f"\n→ Deleting environment '{environment_name}'...")
        print("  This may take a moment...")

        try:
            subprocess.run(['conda', 'env', 'remove', '-n', environment_name, '-y'],
                           check=True, capture_output=True)
            print(f"✓ Environment '{environment_name}' deleted successfully!")
            print("  Storage space has been freed.")

            return True, False, None

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to delete environment: {e}")
            print(f"You may need to delete it manually with: conda env remove -n {environment_name}")
            return False, environment_created_by_notebook, environment_name
    else:
        print("\n✗ Deletion cancelled - environment preserved")
        return False, environment_created_by_notebook, environment_name
