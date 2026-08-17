---
name: moebius-llama
description: Möbius-Llama - Self-reflective transformer architecture that adds internal reasoning loops to any decoder-only LLM. Works with Llama, Mistral, Qwen, Gemma, Phi, Falcon. Golden-ratio decay, adaptive computation, weight-preserving patching.
---

# Möbius-Llama — Self-Reflective Transformers

**Möbius-Llama** is a universal transformer architecture that adds self-reflective reasoning loops to any decoder-only language model. Each layer becomes a loop: Forward → Reflect → Correct → Repeat, with golden-ratio (φ = 1.618...) decay.

> Part of the [PegaConstellation](https://github.com/Immaculate1022/pegaconstellation-hub) / Infinite Optical Fabric ecosystem  
> Free under the **IOF Attribution License v1.0**

## When to Use This

- **Enhancing reasoning** — Adding multi-step reflection to existing LLMs without retraining
- **Building AI assistants** — Creating more thoughtful, self-correcting language models
- **Improving accuracy** — Reducing hallucinations through internal critique loops
- **Deploying efficiently** — Using adaptive computation time for token-level early exit
- **Integrating with any LLM** — Works across Llama, Mistral, Qwen, Gemma, Phi, Falcon, etc.

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

# After `pip install -e .` the package becomes importable:
# from moebius_llama import patch_any_model

# Temporary compatibility path while the full implementation
# is modularized:
from moebius_llama_gift_edition import patch_any_model   # or equivalent entry point

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

model = patch_any_model(model, depth=3, patch_ratio=0.5)

inputs = tokenizer("What is 2+2?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Core Concepts

### Möbius Reflection
Each transformer layer is wrapped in a reflective loop:
1. Forward pass
2. Reflection / critique signal
3. Correction blended with golden-ratio decay
4. Configurable depth (typically 3)

### Golden-Ratio Decay
Reflection magnitude decays by φ^-(i+1). Early loops dominate; later loops refine.

### Universal Adapter
Preserves original layer implementations, monkey-patches `forward()`, detects RoPE / GQA / ALiBi / etc., and keeps pretrained weights intact.

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| Llama 2/3 | ✅ Full | RoPE, GQA, KV-cache |
| Mistral | ✅ Full | Sliding window |
| Qwen2/3 | ✅ Full | Rotary embeddings |
| Gemma / Gemma2 | ✅ Full | RMSNorm native |
| Phi / Phi3 | ✅ Full | Compact architecture |
| Falcon | ✅ Full | Multi-query attention |
| GPT-NeoX | ✅ Full | Parallel attention/MLP |
| MPT | ✅ Full | ALiBi |
| Mamba | ⚠️ Partial | Hybrid SSM/transformer |

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
