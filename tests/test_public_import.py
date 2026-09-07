import torch
from torch import nn

from moebius_llama import patch_any_model


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])


def test_public_patch_import_and_hook():
    model = TinyDecoder()
    returned = patch_any_model(model, depth=1, patch_ratio=1.0)
    assert returned is model
    assert model._moebius_patch_config["layers_patched"] == 2
    assert len(model._moebius_hook_handles) == 2


def test_patch_validates_configuration():
    model = TinyDecoder()
    try:
        patch_any_model(model, depth=0)
    except ValueError as exc:
        assert "depth" in str(exc)
    else:
        raise AssertionError("depth=0 should be rejected")


if __name__ == "__main__":
    test_public_patch_import_and_hook()
    test_patch_validates_configuration()
    print("Möbius public import smoke test: PASS")
