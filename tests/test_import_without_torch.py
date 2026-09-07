from moebius_llama import patch_any_model, unpatch_model


def test_public_api_imports_without_eager_torch_import():
    assert callable(patch_any_model)
    assert callable(unpatch_model)
