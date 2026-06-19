import torch
from typing import Any, Optional


def torch_dtype_from_string(torch_module: Any, dtype_name: Optional[str]) -> Optional[Any]:
    """Convert a string like "bfloat16" or "float32" to the corresponding torch.dtype object."""
    if not hasattr(torch_module, "bfloat16") or not hasattr(torch_module, "float16") or not hasattr(torch_module, "float32"):
        raise RuntimeError("torch module does not have expected dtype attributes. Ensure torch is installed and imported correctly.")
    if dtype_name is None:
        return None

    dtype_name = str(dtype_name).lower()
    mapping = {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "half": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def count_parameters(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total
