"""
Streaming Unsloth SFT trainer for Gemma 4 and Qwen3.5.

Features:
  - argparse + YAML config
  - ClearML logging
  - Unsloth model loading
  - LoRA or full fine-tuning
  - Hugging Face datasets streaming for large JSONL files
  - TRL SFTTrainer with assistant_only_loss=True
  - Conversational JSONL format:
      {"messages": [{"role": "system", "content": "..."}, ...]}

Run:
  python train_streaming_unsloth_sft.py --config config.yaml

Inspect one processed batch before training:
  python train_streaming_unsloth_sft.py --config config.yaml --inspect-batch --dry-run

Override YAML:
  python train_streaming_unsloth_sft.py --config config.yaml \
    --set training.max_steps=20000 \
    --set training.method=full
"""

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import sys
import re
from typing import Any, Dict, Iterable, Optional, Union
from utils import save_outputs, LOG, set_environment
from torch_utils import torch_dtype_from_string, count_parameters
from config import load_config
from data import (
    TokenCountLoggingCollator,
    inspect_first_batch,
    messages_to_prompt_completion,
    messages_to_vlm_prompt_completion,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming Unsloth SFT trainer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config value, e.g. --set training.max_steps=20000",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build model/datasets/trainer but do not train.")
    parser.add_argument(
        "--inspect-batch",
        action="store_true",
        help="Inspect first collated batch and verify assistant-only labels are not all masked.",
    )
    return parser.parse_args()


def filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a kwargs dict to only those accepted by the callable_obj, based on its signature. If the
    callable accepts **kwargs, all non-None values are returned. Otherwise, only the parameters
    explicitly defined in the callable's signature are returned.
    """
    signature = inspect.signature(callable_obj)
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if has_var_kwargs:
        return {k: v for k, v in kwargs.items() if v is not None}
    return {
        k: v
        for k, v in kwargs.items()
        if k in signature.parameters and v is not None
    }


def is_moe_like_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(token in lowered for token in ("a3b", "a4b", "a10b", "a17b", "moe"))


def choose_unsloth_loader(cfg: dict[str, Any], fast_language_model: Any, fast_model: Any) -> Any:
    """
    Determines which Unsloth loader class to use based on config and model name heuristics.
    """
    loader = str(cfg["model"].get("loader", "auto")).lower()
    model_name = cfg["model"]["model_name"]

    if loader == "fast_language_model":
        return fast_language_model
    if loader == "fast_model":
        if fast_model is None:
            raise RuntimeError("unsloth.FastModel is unavailable in this environment.")
        return fast_model
    if loader != "auto":
        raise ValueError("model.loader must be auto, fast_language_model, or fast_model.")

    if is_moe_like_model_name(model_name) and fast_model is not None:
        return fast_model

    return fast_language_model


def init_clearml(cfg: dict[str, Any], config_path: str) -> Optional[Any]:
    if not cfg["clearml"].get("enabled", True):
        return None

    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError("ClearML is enabled but clearml is not installed.") from exc

    seed = cfg["project"].get("seed")
    if seed is not None:
        Task.set_random_seed(int(seed))

    task = Task.init(
        project_name=cfg["clearml"]["project_name"],
        task_name=cfg["clearml"]["task_name"],
        output_uri=cfg["clearml"].get("output_uri"),
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True,
    )

    tags = cfg["clearml"].get("tags") or []
    if tags:
        task.add_tags(tags)

    task.connect_configuration(name="yaml_config_file", configuration=config_path)
    task.connect(copy.deepcopy(cfg), name="resolved_config")
    return task


def get_inner_tokenizer(processing_class: Any) -> Any:
    """
    For VLM processors, return the wrapped tokenizer.
    For normal text models, return the tokenizer itself.
    """
    return getattr(processing_class, "tokenizer", processing_class)


def configure_chat_template(processing_class: Any, cfg: dict[str, Any]) -> Any:
    """
    Ensure tokenizer/processor has a chat_template.
    Needed especially for Gemma 4 processor-based models.
    """
    family = str(cfg["model"].get("family", "")).lower()
    chat_template = cfg["model"].get("chat_template")

    if chat_template is None and family == "gemma4":
        chat_template = "gemma-4"

    if chat_template is None:
        return processing_class

    from unsloth.chat_templates import get_chat_template

    inner = get_inner_tokenizer(processing_class)

    updated_inner = get_chat_template(
        inner,
        chat_template=chat_template,
    )

    # If get_chat_template returned a new tokenizer object, preserve it.
    if hasattr(processing_class, "tokenizer"):
        processing_class.tokenizer = updated_inner
        if getattr(updated_inner, "chat_template", None):
            processing_class.chat_template = updated_inner.chat_template
    else:
        processing_class = updated_inner

    if not getattr(processing_class, "chat_template", None):
        raise RuntimeError(
            f"No chat_template is set after applying {chat_template!r}. "
            "Check model.chat_template and the loaded tokenizer/processor type."
        )

    LOG.info("Applied chat template: %s", chat_template)
    return processing_class


def configure_tokenizer(processing_class: Any, cfg: dict[str, Any]) -> Any:
    """
    Configure EOS/PAD tokens on the actual tokenizer.
    Works for both normal tokenizers and processor-wrapped tokenizers.
    """
    tokenizer = get_inner_tokenizer(processing_class)

    family = str(cfg["model"].get("family", "")).lower()
    eos_setting = cfg["model"].get("eos_token", "auto")
    pad_setting = cfg["model"].get("pad_token", "auto")

    if eos_setting and eos_setting != "auto":
        tokenizer.eos_token = eos_setting
    elif eos_setting == "auto" and family.startswith("qwen"):
        try:
            token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if token_id is not None and token_id != tokenizer.unk_token_id:
                tokenizer.eos_token = "<|im_end|>"
                LOG.info("Set Qwen EOS token to <|im_end|>.")
        except Exception as exc:
            LOG.warning("Could not auto-set Qwen EOS token; using tokenizer default. Error: %s", exc)

    if pad_setting and pad_setting != "auto":
        tokenizer.pad_token = pad_setting
    elif pad_setting == "auto" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOG.info("Set pad_token to eos_token.")

    # Mirror fields onto the processor when possible.
    if hasattr(processing_class, "tokenizer"):
        for attr in ("eos_token", "eos_token_id", "pad_token", "pad_token_id"):
            if hasattr(tokenizer, attr):
                try:
                    setattr(processing_class, attr, getattr(tokenizer, attr))
                except Exception:
                    pass

    LOG.info("tokenizer type: %s", type(tokenizer))
    LOG.info("eos_token: %r / %r", getattr(tokenizer, "eos_token", None), getattr(tokenizer, "eos_token_id", None))
    LOG.info("pad_token: %r / %r", getattr(tokenizer, "pad_token", None), getattr(tokenizer, "pad_token_id", None))
    LOG.info("chat_template set: %s", bool(getattr(processing_class, "chat_template", None)))

    return processing_class
        


def maybe_take(dataset: Any, n: Optional[int], streaming: bool) -> Any:
    """If n is not None, return a dataset with at most n examples. Uses streaming-friendly .take() if streaming, otherwise .select()."""
    if n is None:
        return dataset
    n = int(n)
    if n <= 0:
        return dataset.take(0) if streaming else dataset.select(range(0))
    if streaming:
        return dataset.take(n)
    return dataset.select(range(min(n, len(dataset))))


def get_dataset_mapper(cfg: dict[str, Any]) -> Any:
    data_format = str(cfg["data"].get("format", "text_prompt_completion")).strip().lower()
    messages_field = cfg["data"].get("messages_field", "messages")
    do_validate = bool(cfg["data"].get("validate_messages", True))

    if data_format == "text_prompt_completion":
        return lambda ex: messages_to_prompt_completion(ex, messages_field, do_validate)

    if data_format == "vlm_prompt_completion":
        image_root = cfg["data"].get("image_root")
        return lambda ex: messages_to_vlm_prompt_completion(
            ex,
            messages_field,
            do_validate,
            image_root=image_root,
        )

    raise ValueError(
        "data.format must be one of: text_prompt_completion, vlm_prompt_completion. "
        f"Got {data_format!r}."
    )


def load_one_split(cfg: dict[str, Any], split_name: str, data_files: Union[str, list[str]]) -> Any:
    """Load one dataset split from JSONL files, with optional streaming and shuffling."""
    from datasets import load_dataset

    streaming = bool(cfg["data"].get("streaming", True))
    mapper = get_dataset_mapper(cfg)

    LOG.info("Loading %s split from %s with streaming=%s", split_name, data_files, streaming)

    dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=streaming,
    )

    if split_name == "train" and cfg["data"].get("shuffle_train", True):
        if not streaming:
            dataset = dataset.shuffle(seed=int(cfg["project"].get("seed", 3407)))
        else:
            dataset = dataset.shuffle(
                seed=int(cfg["project"].get("seed", 3407)),
                buffer_size=int(cfg["data"].get("shuffle_buffer_size", 10000)),
            )

    if split_name == "train":
        dataset = maybe_take(dataset, cfg["data"].get("max_train_samples"), streaming)
    else:
        dataset = maybe_take(dataset, cfg["data"].get("max_eval_samples"), streaming)

    dataset = dataset.map(mapper)

    return dataset


def load_streaming_datasets(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load training and validation datasets with optional streaming."""
    train_dataset = load_one_split(cfg, "train", cfg["data"]["train_jsonl"])

    eval_dataset = None
    validation_jsonl = cfg["data"].get("validation_jsonl")
    if validation_jsonl and str(cfg["training"].get("eval_strategy", "no")).lower() != "no":
        eval_dataset = load_one_split(cfg, "validation", validation_jsonl)

    return {
        "train": train_dataset,
        "validation": eval_dataset,
    }


def load_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Load a model and its tokenizer based on the provided configuration."""
    # Import Unsloth before importing TRL/Transformers training components.
    from unsloth import FastLanguageModel

    try:
        from unsloth import FastModel
    except Exception:
        FastModel = None

    import torch

    loader_cls = choose_unsloth_loader(cfg, FastLanguageModel, FastModel)
    dtype = torch_dtype_from_string(torch, cfg["model"].get("dtype"))

    load_kwargs = {
        "model_name": cfg["model"]["model_name"],
        "max_seq_length": int(cfg["model"]["max_seq_length"]),
        "dtype": dtype,
        "trust_remote_code": cfg["model"].get("trust_remote_code"),
        "device_map": cfg["model"].get("device_map"),
        "fast_inference": cfg["model"].get("fast_inference"),
        "load_in_4bit": cfg["model"].get("load_in_4bit"),
        "load_in_8bit": cfg["model"].get("load_in_8bit"),
        "load_in_16bit": cfg["model"].get("load_in_16bit"),
        "full_finetuning": cfg["model"].get("full_finetuning"),
    }

    LOG.info("Loading model %s via %s", cfg["model"]["model_name"], loader_cls.__name__)
    model, tokenizer = loader_cls.from_pretrained(
        **filter_kwargs(loader_cls.from_pretrained, load_kwargs)
    )

    tokenizer = configure_chat_template(tokenizer, cfg)
    tokenizer = configure_tokenizer(tokenizer, cfg)


    if cfg["training"]["method"] == "lora":
        lora_kwargs = {
            "r": cfg["lora"].get("r"),
            "target_modules": cfg["lora"].get("target_modules"),
            "lora_alpha": cfg["lora"].get("lora_alpha"),
            "lora_dropout": cfg["lora"].get("lora_dropout"),
            "bias": cfg["lora"].get("bias"),
            "use_gradient_checkpointing": cfg["lora"].get("use_gradient_checkpointing"),
            "random_state": cfg["lora"].get("random_state", cfg["project"].get("seed")),
            "max_seq_length": cfg["model"].get("max_seq_length"),
            "use_rslora": cfg["lora"].get("use_rslora"),
            "loftq_config": cfg["lora"].get("loftq_config"),
            "modules_to_save": cfg["lora"].get("modules_to_save"),
        }

        LOG.info("Attaching LoRA adapters.")
        model = loader_cls.get_peft_model(
            model,
            **filter_kwargs(loader_cls.get_peft_model, lora_kwargs),
        )
    else:
        LOG.info("Full fine-tuning enabled. No LoRA adapters attached.")

    if hasattr(loader_cls, "for_training"):
        maybe_model = loader_cls.for_training(model)
        if maybe_model is not None:
            model = maybe_model

    trainable, total = count_parameters(model)
    pct = 100.0 * trainable / total if total else 0.0
    LOG.info("Trainable parameters: %d / %d (%.4f%%)", trainable, total, pct)

    LOG.info("tokenizer type: %s", type(tokenizer))
    LOG.info("tokenizer.chat_template exists: %s", bool(getattr(tokenizer, "chat_template", None)))
    LOG.info("processor.chat_template exists: %s", bool(getattr(tokenizer, "chat_template", None)))
    LOG.info("model config model_type: %s", getattr(model.config, "model_type", None))
    LOG.info("model config architectures: %s", getattr(model.config, "architectures", None))

    return model, tokenizer, loader_cls


def instantiate_sft_config(SFTConfig: Any, kwargs: dict[str, Any]) -> Any:
    """Instantiate an SFTConfig object, dropping unsupported arguments."""
    kwargs = dict(kwargs)

    while True:
        try:
            return SFTConfig(**kwargs)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not match:
                raise

            bad_key = match.group(1)
            if bad_key not in kwargs:
                raise

            LOG.warning("Dropping unsupported SFTConfig argument: %s", bad_key)
            kwargs.pop(bad_key)


def make_sft_config(cfg: Dict[str, Any], tokenizer: Any, SFTConfig: Any) -> Any:
    """Create an SFTConfig object from the training config, applying necessary transformations and filtering.
    This function handles mapping config keys to the appropriate SFTConfig parameters, applying defaults, 
    and dropping unsupported parameters based on the SFTConfig signature. 
    It also includes workarounds for specific model requirements, 
    such as ensuring the correct EOS token is set and adjusting for different SFTConfig versions. 

    The resulting SFTConfig object is ready to be passed to the SFTTrainer.
    """
    training_cfg = copy.deepcopy(cfg["training"])

    training_cfg.pop("resume_from_checkpoint", None)
    training_cfg.pop("method", None)
    training_cfg.pop("truncation_mode", None)
    training_cfg.pop("dataset_text_field", None)

    max_seq_length = int(cfg["model"]["max_seq_length"])

    # Important workaround for Qwen3.5 VLM detection.
    training_cfg["assistant_only_loss"] = False
    training_cfg["completion_only_loss"] = True

    if getattr(tokenizer, "eos_token", None):
        training_cfg["eos_token"] = tokenizer.eos_token

    sft_params = inspect.signature(SFTConfig.__init__).parameters

    if "max_length" in sft_params:
        training_cfg["max_length"] = max_seq_length
    elif "max_seq_length" in sft_params:
        training_cfg["max_seq_length"] = max_seq_length

    if "eval_strategy" in training_cfg and "eval_strategy" not in sft_params and "evaluation_strategy" in sft_params:
        training_cfg["evaluation_strategy"] = training_cfg.pop("eval_strategy")
    elif "evaluation_strategy" in training_cfg and "evaluation_strategy" not in sft_params and "eval_strategy" in sft_params:
        training_cfg["eval_strategy"] = training_cfg.pop("evaluation_strategy")

    filtered = filter_kwargs(SFTConfig.__init__, training_cfg)

    # Defensive drops for Unsloth compiled wrappers.
    filtered.pop("truncation_mode", None)

    return instantiate_sft_config(SFTConfig, filtered)


def make_clearml_callback(task: Any):
    """Create a ClearML callback for logging trainer metrics to the ClearML dashboard."""
    from transformers import TrainerCallback

    class ClearMLScalarCallback(TrainerCallback):
        def __init__(self, clearml_task: Any):
            self.task = clearml_task
            self.logger = clearml_task.get_logger()

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            step = int(state.global_step)
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    self.logger.report_scalar(
                        title="trainer",
                        series=str(key),
                        value=float(value),
                        iteration=step,
                    )

    return ClearMLScalarCallback(task)


def build_trainer(
    cfg: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    datasets: Dict[str, Any],
    clearml_task: Optional[Any],
) -> Any:
    """Build and return an SFTTrainer based on the provided model, tokenizer, datasets, and configuration."""
    from trl import SFTConfig, SFTTrainer

    sft_args = make_sft_config(cfg, tokenizer, SFTConfig)

    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": datasets["train"],
    }

    if datasets.get("validation") is not None:
        trainer_kwargs["eval_dataset"] = datasets["validation"]

    sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    if clearml_task is not None:
        trainer.add_callback(make_clearml_callback(clearml_task))

    if cfg.get("logging", {}).get("log_token_counts", True):
        trainer.data_collator = TokenCountLoggingCollator(
            base_collator=trainer.data_collator,
            clearml_task=clearml_task,
            log_every=int(cfg.get("logging", {}).get("token_count_log_every", 100)),
            log_prefix="tokens/train",
        )

    return trainer


def main() -> None:
    setup_logging()
    args = parse_args()

    cfg = load_config(args.config, args.overrides)
    set_environment(cfg)

    clearml_task = init_clearml(cfg, args.config)

    model, tokenizer, _loader_cls = load_model_and_tokenizer(cfg)

    from transformers import set_seed

    if cfg["project"].get("seed") is not None:
        set_seed(int(cfg["project"]["seed"]))

    datasets = load_streaming_datasets(cfg)
    trainer = build_trainer(cfg, model, tokenizer, datasets, clearml_task)

    if args.inspect_batch:
        inspect_first_batch(trainer, tokenizer)

    if args.dry_run:
        LOG.info("Dry run complete. Exiting before training.")
        if clearml_task is not None:
            clearml_task.close()
        return

    LOG.info("Starting training.")
    train_result = trainer.train(
        resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint")
    )

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if datasets.get("validation") is not None and str(cfg["training"].get("eval_strategy", "no")).lower() != "no":
        LOG.info("Running final evaluation.")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    save_outputs(cfg, model, tokenizer, trainer, clearml_task)

    if clearml_task is not None:
        clearml_task.close()

    LOG.info("Done.")


if __name__ == "__main__":
    main()
