"""Transformer-based multi-task multi-label text classifier.

Supports any HuggingFace AutoModel-compatible backbone, including all
Czech-capable models listed in task.md.  Custom PyTorch modules can also be
injected directly via ``TransformerClassifier.from_custom_encoder``.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)


def _patch_missing_post_init(config: Any) -> None:
    """Patch remote-code model classes that omit ``self.post_init()`` in their
    ``__init__``, and apply other known PyTorch/transformers compatibility fixes.

    Since transformers 5.x, ``post_init()`` must be called at the end of every
    ``PreTrainedModel.__init__`` to populate ``self.all_tied_weights_keys``.
    Older community / Hub models (e.g. ``LtgbertModel``) were written before
    this requirement and therefore crash during ``from_pretrained``.

    Additionally patches:
    - ``MaskedSoftmax.backward``: uses ``softmax_backward_data`` which was
      removed in PyTorch 2.x.  Replaced with the equivalent expression
      ``output * (grad - sum(grad * output, dim))``.

    The function loads each module via ``get_class_in_module`` (the same path
    used by ``AutoModel.from_pretrained``), which records
    ``__transformers_module_hash__``.  The subsequent call from
    ``from_pretrained`` therefore finds the already-patched cached module
    rather than reloading from disk.
    """
    from transformers.dynamic_module_utils import get_cached_module_file, get_class_in_module

    auto_map = getattr(config, "auto_map", {})
    if not auto_map:
        return

    model_id: str = config._name_or_path
    patched_mod_keys: set[str] = set()

    for ref in auto_map.values():
        if "." not in ref:
            continue
        mod_file, cls_name = ref.rsplit(".", 1)
        try:
            final_module = get_cached_module_file(model_id, mod_file + ".py")
            cls = get_class_in_module(cls_name, final_module)
        except Exception:
            continue

        # --- Module-level patches (applied once per unique module file) ---
        mod_key = os.path.normpath(final_module).removesuffix(".py").replace(os.sep, ".")
        if mod_key not in patched_mod_keys:
            patched_mod_keys.add(mod_key)
            mod = sys.modules.get(mod_key)
            if mod is not None:
                _patch_masked_softmax_backward(mod)

        # --- Class-level patch: missing post_init() ---
        try:
            src = inspect.getsource(cls.__init__)
        except (OSError, TypeError):
            continue
        if "post_init" not in src:
            logger.debug(
                "Patching %s.__init__ to call post_init() (transformers >=5 compat)",
                cls_name,
            )
            _orig = cls.__init__

            def _patched(self, *args, __orig=_orig, **kwargs):
                __orig(self, *args, **kwargs)
                if not hasattr(self, "all_tied_weights_keys"):
                    self.post_init()

            cls.__init__ = _patched


def _patch_masked_softmax_backward(mod: Any) -> None:
    """Replace ``MaskedSoftmax.backward`` that calls the removed
    ``softmax_backward_data`` with the equivalent analytical expression.

    ``softmax_backward_data`` was a C-extension helper in PyTorch < 2.x.
    The gradient of a masked softmax is identical to the standard softmax
    gradient:  ``d_x = output * (d_y - sum(d_y * output, dim))``.
    Masked positions have ``output == 0`` so they receive zero gradient
    without any extra masking step.
    """
    import torch

    masked_softmax_cls = getattr(mod, "MaskedSoftmax", None)
    if masked_softmax_cls is None:
        return
    try:
        src = inspect.getsource(masked_softmax_cls.backward)
    except (OSError, TypeError):
        return
    if "softmax_backward_data" not in src:
        return

    logger.debug("Patching MaskedSoftmax.backward to remove softmax_backward_data (PyTorch >=2 compat)")

    @staticmethod  # type: ignore[misc]
    def _fixed_backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        (output,) = ctx.saved_tensors
        input_grad = output * (
            grad_output - (grad_output * output).sum(dim=ctx.dim, keepdim=True)
        )
        return input_grad, None, None

    masked_softmax_cls.backward = _fixed_backward


class ClassificationHead(nn.Module):
    """Single-task linear classification head with dropout."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(hidden))


class TransformerClassifier(nn.Module):
    """Multi-task multi-label classifier with a shared pretrained transformer encoder.

    Each task gets its own ``ClassificationHead``.  The encoder is loaded from
    HuggingFace Hub via ``AutoModel`` and can be any BERT-style model.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tasks: Ordered list of task names.
        num_classes: Mapping from task name to number of output classes.
        dropout: Dropout probability applied before each classification head.
        pooling: Pooling strategy over token representations.
            ``"cls"`` uses the first token; ``"mean"`` uses masked mean pooling.
        freeze_encoder_layers: Number of initial encoder layers to freeze (0 = none).
    """

    def __init__(
        self,
        model_name_or_path: str,
        tasks: list[str],
        num_classes: dict[str, int],
        dropout: float = 0.1,
        pooling: str = "cls",
        freeze_encoder_layers: int = 0,
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.pooling = pooling

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        _patch_missing_post_init(config)
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(
            model_name_or_path, config=config, trust_remote_code=True
        )
        # Encoder-decoder models (e.g. BART): route forward() through the
        # encoder sub-module only to avoid wasting memory/compute on the decoder.
        self._is_encoder_decoder: bool = getattr(config, "is_encoder_decoder", False)
        # BART-style configs expose hidden size as `d_model` rather than `hidden_size`.
        hidden_size: int = (
            getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        )

        self.heads = nn.ModuleDict(
            {task: ClassificationHead(hidden_size, num_classes[task], dropout) for task in tasks}
        )

        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)

    @classmethod
    def from_custom_encoder(
        cls,
        encoder: nn.Module,
        hidden_size: int,
        tasks: list[str],
        num_classes: dict[str, int],
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> "TransformerClassifier":
        """Create a classifier from an arbitrary PyTorch encoder module.

        The encoder must accept ``(input_ids, attention_mask)`` and return an
        object with a ``last_hidden_state`` attribute of shape (B, T, H).
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.tasks = tasks
        obj.pooling = pooling
        obj.encoder = encoder
        obj._is_encoder_decoder = False
        obj.heads = nn.ModuleDict(
            {task: ClassificationHead(hidden_size, num_classes[task], dropout) for task in tasks}
        )
        return obj

    def _freeze_encoder_layers(self, n: int) -> None:
        """Freeze the first *n* encoder layers (BERT-style architecture)."""
        encoder_layers = None
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            encoder_layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "roberta") and hasattr(self.encoder.roberta, "encoder"):
            encoder_layers = self.encoder.roberta.encoder.layer

        if encoder_layers is None:
            logger.warning(
                "Could not locate encoder layers to freeze for this model architecture."
            )
            return

        frozen = 0
        for i, layer in enumerate(encoder_layers):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen += 1
        logger.info("Froze %d encoder layers", frozen)

    def _pool(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]
        elif self.pooling == "mean":
            token_emb = outputs.last_hidden_state  # (B, T, H)
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling!r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run encoder and all classification heads.

        Returns:
            Dict mapping task name to raw logits of shape (B, num_classes[task]).
        """
        encoder_kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        if self._is_encoder_decoder:
            # Use only the encoder to get contextualised representations;
            # avoids materialising decoder states which are irrelevant for
            # classification and waste both memory and compute.
            encoder_component = self.encoder.get_encoder()
            outputs = encoder_component(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(**encoder_kwargs)
        pooled = self._pool(outputs, attention_mask)

        return {task: self.heads[task](pooled) for task in self.tasks}
