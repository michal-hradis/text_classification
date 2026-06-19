import logging
LOG = logging.getLogger("streaming_unsloth_sft")

from typing import Any, Dict, Optional
import os


def save_outputs(cfg: Dict[str, Any], model: Any, tokenizer: Any, trainer: Any, clearml_task: Optional[Any]) -> None:
    """
    Save the outputs of the training process, including the final model, merged 16-bit model, and GGUF model.
    """
    output_dir = Path(cfg["training"]["output_dir"])
    save_cfg = cfg.get("save") or {}

    if save_cfg.get("save_final", True):
        final_dir = output_dir / save_cfg.get("final_dir_name", "final")
        final_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("Saving final trainer/model output to %s", final_dir)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

    if save_cfg.get("save_merged_16bit", False):
        merged_dir = output_dir / save_cfg.get("merged_16bit_dir_name", "merged_16bit")
        merged_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained_merged"):
            LOG.info("Saving merged 16-bit model to %s", merged_dir)
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
        else:
            LOG.warning("save_pretrained_merged is unavailable; skipping merged save.")

    if save_cfg.get("save_gguf", False):
        gguf_dir = output_dir / save_cfg.get("gguf_dir_name", "gguf")
        gguf_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained_gguf"):
            quant = save_cfg.get("gguf_quantization_method", "q8_0")
            LOG.info("Saving GGUF to %s with quantization=%s", gguf_dir, quant)
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method=quant,
            )
        else:
            LOG.warning("save_pretrained_gguf is unavailable; skipping GGUF save.")

    if clearml_task is not None:
        clearml_task.upload_artifact("resolved_config", cfg)


def set_environment(cfg: Dict[str, Any]) -> None:
    """Set environment variables from config. For example, this is used to set CUDA_VISIBLE_DEVICES."""
    for key, value in (cfg.get("env") or {}).items():
        if value is not None:
            os.environ[str(key)] = str(value)