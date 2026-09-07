"""Compatibility entry point for the Möbius-Llama research adapter.

The maintained implementation lives in :mod:`moebius_llama`. Keeping this
small module importable avoids duplicate experimental implementations drifting
out of sync.
"""

from moebius_llama import patch_any_model, unpatch_model


def patch_to_moebius(model, depth=3, patch_ratio=0.5, **_legacy_options):
    """Apply the maintained adapter using the historical function name."""
    return patch_any_model(model, depth=depth, patch_ratio=patch_ratio)


__all__ = ["patch_any_model", "patch_to_moebius", "unpatch_model"]
