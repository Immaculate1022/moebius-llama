"""
Möbius-Llama — Self-reflective transformer architecture
for any decoder-only LLM.

Part of the PegaConstellation / Infinite Optical Fabric ecosystem.
Licensed under IOF Attribution License v1.0
"""

__version__ = "0.1.0"
__author__ = "Gregory Scott Davis"

from .adapter import patch_any_model

__all__ = [
    "__version__",
    "patch_any_model",
]
