"""Experimental, weight-preserving reflection hooks for decoder-only models.

This module intentionally avoids importing Transformers at package-import time.
It supports common layer containers used by decoder-only architectures and leaves
models unchanged when no supported container is found.
"""
from __future__ import annotations

from typing import Any
import math


_LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "decoder.layers",
)


def _resolve_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def _replace_hidden(output: Any, depth: int, patch_ratio: float) -> Any:
    """Apply a bounded, non-parametric reflection to a layer output."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency is optional at import time
        raise RuntimeError("patch_any_model requires torch to be installed") from exc

    if isinstance(output, dict) and "last_hidden_state" in output:
        hidden = output["last_hidden_state"]
    else:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
        return output

    reflected = hidden
    phi = (1.0 + 5.0**0.5) / 2.0
    for index in range(max(1, depth)):
        center = reflected.mean(dim=-1, keepdim=True)
        reflected = reflected + ((center - reflected) * (patch_ratio * phi ** (-(index + 1))))
    if isinstance(output, dict) and "last_hidden_state" in output:
        updated = output.copy()
        updated["last_hidden_state"] = reflected
        return updated
    updated = (reflected, *output[1:]) if isinstance(output, tuple) else reflected
    if isinstance(output, list):
        updated = [reflected, *output[1:]]
    return updated


def patch_any_model(model: Any, depth: int = 3, patch_ratio: float = 0.5) -> Any:
    """Attach experimental reflection hooks to common decoder-only layer stacks.

    The function returns the same model instance. It does not replace weights or
    claim production compatibility. Hooks are attached to approximately the first
    ``patch_ratio`` of the discovered layer stack; ``depth`` controls the bounded
    reflection passes applied to each layer output.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if not 0.0 <= patch_ratio <= 1.0:
        raise ValueError("patch_ratio must be between 0 and 1")
    if not hasattr(model, "register_buffer"):
        raise TypeError("model must be a torch.nn.Module-like object")

    layers = None
    for path in _LAYER_PATHS:
        candidate = _resolve_path(model, path)
        if candidate is not None and hasattr(candidate, "__len__") and hasattr(candidate, "__getitem__"):
            layers = candidate
            break
    if layers is None:
        raise ValueError("could not find a supported decoder layer stack")

    count = len(layers)
    selected = min(count, math.ceil(count * patch_ratio)) if patch_ratio else 0
    handles = []
    for layer in list(layers)[:selected]:
        handles.append(layer.register_forward_hook(lambda _module, _inputs, output: _replace_hidden(output, depth, patch_ratio)))

    unpatch_model(model)
    model._moebius_hook_handles = handles
    model._moebius_patch_config = {"depth": depth, "patch_ratio": patch_ratio, "layers_patched": selected}
    return model


def unpatch_model(model: Any) -> Any:
    """Remove hooks installed by :func:`patch_any_model` and return ``model``."""
    for handle in getattr(model, "_moebius_hook_handles", []):
        handle.remove()
    model._moebius_hook_handles = []
    if hasattr(model, "_moebius_patch_config"):
        delattr(model, "_moebius_patch_config")
    return model


__all__ = ["patch_any_model", "unpatch_model"]
