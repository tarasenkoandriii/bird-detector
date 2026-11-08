"""Shared paths and device selection. Import torch only after MPS setup."""

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent


def resolve_device(requested):
    import torch

    torch.set_num_threads(min(6, os.cpu_count() or 1))
    if requested == "auto":
        return (
            "mps"
            if torch.backends.mps.is_available()
            else ("0" if torch.cuda.is_available() else "cpu")
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu or --device auto.")
    return requested
