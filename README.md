---
name: moebius-llama
description: Möbius-Llama - Experimental bounded reflection hooks for supported decoder-only transformer layer stacks. Golden-ratio decay and weight-preserving patching.
---

# Möbius-Llama — Self-Reflective Transformers

**Möbius-Llama** is an experimental adapter that applies bounded reflection hooks to supported decoder-only transformer layer stacks. It is research code, not a published model, production compatibility guarantee, or benchmarked improvement. The current package preserves the original weights and exposes small, explicit `patch_any_model` and `unpatch_model` entry points.

> Part of the [PegaConstellation](https://github.com/Immaculate1022/pegaconstellation-hub) / Infinite Optical Fabric ecosystem  
> Free under the **IOF Attribution License v1.0**

## When to Use This

- **Enhancing reasoning** — Adding multi-step reflection to existing LLMs without retraining
- **Building AI assistants** — Creating more thoughtful, self-correcting language models
- **Improving accuracy** — Reducing hallucinations through internal critique loops
- **Integrating with supported decoder stacks** — Common Llama, Mistral, Qwen, Gemma, Phi, and Falcon layouts are discoverable, but each model should be tested before use.

## Installation

### Recommended (editable / development)

```bash
git clone https://github.com/Immaculate1022/moebius-llama.git
cd moebius-llama
pip install -e .
# or with uv (fast)
uv pip install -e .
```

### From source without packaging

```bash
git clone https://github.com/Immaculate1022/moebius-llama.git
cd moebius-llama
pip install torch transformers accelerate peft
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from moebius_llama import patch_any_model, unpatch_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

model = patch_any_model(model, depth=3, patch_ratio=0.5)

# Remove the experimental hooks when the comparison is complete.
# unpatch_model(model)

inputs = tokenizer("What is 2+2?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Core Concepts

### Möbius Reflection
Each selected transformer layer receives a bounded output transformation:
1. Forward pass
2. Hidden-state centering
3. Correction blended with golden-ratio decay
4. Configurable depth (typically 3)

### Golden-Ratio Decay
Reflection magnitude decays by φ^-(i+1). Early loops dominate; later loops refine.

### Universal Adapter
Preserves original layer implementations, attaches bounded output hooks, leaves model-specific attention and positional logic untouched, and keeps pretrained weights intact.

## Current package boundary

The package currently discovers common layer containers (`model.layers`, `transformer.h`, `gpt_neox.layers`, `model.decoder.layers`, and `decoder.layers`) and attaches bounded output hooks to a selected fraction of those layers. Call `unpatch_model(model)` to remove the hooks and restore the original execution path. This is a research scaffold for controlled experiments. It does not establish full compatibility with every model family, preserve generation quality, or demonstrate a reasoning or accuracy advantage. The supported layer paths and output shapes are intentionally conservative, so unsupported model structures fail clearly instead of being patched silently.

## Integration with PegaConstellation

Möbius-Llama serves as the AI backbone for:

- **Aetherius Nexus** — Enhanced research assistant
- **AHR-Endpoint** — Intelligent threat analysis
- **IOF Design Grammar** — Meta-reasoning about system design

Related geometry substrate: [Tesseract Medium](https://github.com/Immaculate1022/tesseract-medium)

## License

**IOF Attribution License v1.0**  
Free for development, implementation, research, and AI training.  
Attribution required for public distribution or derivatives:

> Möbius-Llama by Gregory Scott Davis, Princeton, NC.

---

**PegaConstellation · Gregory Scott Davis**  
*Princeton, NC*
