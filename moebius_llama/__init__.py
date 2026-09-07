"""
Möbius-Llama — Experimental bounded reflection hooks
for supported decoder-only transformer layer stacks.

Part of the PegaConstellation / Infinite Optical Fabric ecosystem.
Licensed under IOF Attribution License v1.0
"""

__version__ = "0.1.0"
__author__ = "Gregory Scott Davis"

from .adapter import patch_any_model, unpatch_model

__all__ = [
    "__version__",
    "patch_any_model",
    "unpatch_model",
]
