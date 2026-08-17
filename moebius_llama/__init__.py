"""
Möbius-Llama — Self-reflective transformer architecture
for any decoder-only LLM.

Part of the PegaConstellation / Infinite Optical Fabric ecosystem.
Licensed under IOF Attribution License v1.0
"""

__version__ = "0.1.0"
__author__ = "Gregory Scott Davis"

# Compatibility import path.
# The full implementation currently lives in the gift-edition script.
# This package structure enables clean `from moebius_llama import ...`
# and future modularization without breaking existing users.

try:
    # Prefer the package-local modules once they exist
    from .adapter import patch_any_model  # type: ignore
except ImportError:
    # Fallback: users can still import from the original script
    # or place the implementation under moebius_llama/
    patch_any_model = None  # type: ignore

__all__ = [
    "__version__",
    "patch_any_model",
]
