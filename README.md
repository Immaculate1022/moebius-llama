---
name: moebius-llama
description: Experimental, weight-preserving reflection hooks for supported decoder-only transformer layer stacks.
---

# Möbius-Llama — Self-Reflective Transformers

**Möbius-Llama** is a small, experimental Python adapter for researchers and developers who want to test bounded reflection hooks on a compatible decoder-only Transformer. It attaches forward hooks to a portion of a model’s decoder layers; it does not train a model or replace its pretrained weights.

To try it, install the package from this repository, load a compatible model with Transformers, and call `patch_any_model`. Remove the hooks with `unpatch_model` when your comparison is finished.

> **Status: alpha research code.** Möbius-Llama is not a published model, a production compatibility guarantee, or a benchmarked improvement. Evaluate its effect on your own model and workload before relying on it.

## Quick start

Möbius-Llama requires **Python 3.10+**. Installing the project also installs its declared PyTorch and Hugging Face dependencies.

```bash
git clone https://github.com/Immaculate1022/moebius-llama.git
cd moebius-llama
python -m pip install -e .
```

Choose a decoder-only model that you are permitted to download and whose layer stack is supported (see [Compatibility and limitations](#compatibility-and-limitations)). Then patch it in place:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from moebius_llama import patch_any_model, unpatch_model

model_id = "<a compatible decoder-only Transformers model>"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# This returns the same model instance with experimental forward hooks attached.
model = patch_any_model(model, depth=3, patch_ratio=0.5)

inputs = tokenizer("What is 2 + 2?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))

# Remove Möbius-Llama hooks before an unpatched comparison or when finished.
unpatch_model(model)
```

## What the adapter does

For each selected decoder layer, Möbius-Llama registers a forward hook that applies a non-parametric reflection to the layer’s hidden-state output. The hook repeatedly moves values toward their mean over the final hidden dimension. The correction decays by the golden ratio over the requested reflection passes.

`depth` sets the number of reflection passes and must be at least `1`. `patch_ratio` must be between `0` and `1`; it selects approximately that fraction of the **first** layers in the discovered stack and also sets the correction scale. For example, a `patch_ratio` of `0.5` selects `ceil(half the layers)` and uses `0.5` as the scale factor.

The adapter preserves pretrained weights and returns the same model instance. `unpatch_model(model)` removes hooks installed by Möbius-Llama and clears its recorded patch configuration.

## Compatibility and limitations

The package looks for the first available layer container at one of these paths:

- `model.layers`
- `transformer.h`
- `gpt_neox.layers`
- `model.decoder.layers`
- `decoder.layers`

This covers several common decoder-only layouts, but it is not a claim of full support for every model family or checkpoint. If no listed layer stack is found, `patch_any_model` raises a `ValueError` rather than patching a model silently. A compatible layer must also produce a tensor, tuple/list whose first item is a tensor, or a dictionary with `last_hidden_state` for the hook to change its output.

Möbius-Llama has no included quality, reasoning, accuracy, or hallucination benchmark. It does not guarantee generation quality, performance, or compatibility with a particular architecture. Treat it as a controlled experiment: test patched and unpatched models with the same configuration, and keep the unpatched baseline.

## Public API

```python
from moebius_llama import patch_any_model, unpatch_model
```

- `patch_any_model(model, depth=3, patch_ratio=0.5)` attaches the experimental hooks and returns `model`.
- `unpatch_model(model)` removes hooks previously installed by the adapter and returns `model`.

## Project links

- [Repository](https://github.com/Immaculate1022/moebius-llama)
- [PegaConstellation Hub](https://github.com/Immaculate1022/pegaconstellation-hub), listed in the package metadata.

## License

Möbius-Llama is licensed under the [IOF Attribution License v1.0](LICENSE).

The license permits use, copying, modification, publication, distribution, sublicensing, and deployment for any purpose. Any public use, derivative work, or implementation must include clear attribution to:

> Möbius-Llama by Gregory Scott Davis, Princeton, NC.

The software is provided **“AS IS,” without warranty of any kind**.

---

**PegaConstellation · Gregory Scott Davis**  
*Princeton, NC*
