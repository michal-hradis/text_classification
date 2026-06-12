#!/usr/bin/env python3
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
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import yaml


LOG = logging.getLogger("streaming_unsloth_sft")


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "llm-sft",
        "run_name": "streaming-unsloth-sft",
        "seed": 3407,
    },
    "clearml": {
        "enabled": True,
        "project_name": "LLM/SFT",
        "task_name": None,
        "output_uri": None,
        "tags": [],
    },
    "env": {},

    "model": {
        "family": "qwen3_5",  # qwen3_5, gemma4
        "model_name": "unsloth/Qwen3.5-4B",
        "loader": "auto",  # auto, fast_language_model, fast_model
        "max_seq_length": 2048,
        "dtype": "bfloat16",  # bfloat16, float16, float32, null
        "trust_remote_code": True,
        "device_map": None,
        "fast_inference": None,

        # Normalized from training.method.
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,

        "eos_token": "auto",
        "pad_token": "auto",
    },

    "data": {
        "streaming": True,

        # String or list of JSONL files. Lists are useful for sharded >10 GB datasets.
        "train_jsonl": "data/train.jsonl",
        "validation_jsonl": "data/validation.jsonl",

        # Your file already uses "messages".
        # If not, this script maps data.messages_field -> "messages".
        "messages_field": "messages",

        "validate_messages": True,

        # Streaming shuffle is approximate and buffer-based.
        "shuffle_train": True,
        "shuffle_buffer_size": 10000,

        # For huge validation files, keep this bounded.
        # Set null to evaluate the entire validation stream.
        "max_train_samples": None,
        "max_eval_samples": 5000,
    },

    "training": {
        "method": "lora",  # lora or full

        "output_dir": "outputs/streaming-unsloth-sft",

        # Required for streaming datasets because IterableDataset has no length.
        "max_steps": 10000,
        "num_train_epochs": 1,

        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,

        "learning_rate": 2.0e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",

        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "save_total_limit": 3,

        "optim": "adamw_8bit",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,

        # Keep packing off for classification unless you verify masking behavior.
        "packing": False,

        # This is the key change.
        "assistant_only_loss": True,

        # TRL can set or patch templates through this.
        # Usually leave null for Qwen3.5; TRL recognizes Qwen3.5 training templates.
        # For Gemma 4, run --inspect-batch to verify nonzero supervised labels.
        "chat_template_path": None,

        # Use null/[] to avoid duplicate external logging. ClearML is handled manually.
        "report_to": [],

        # Optional checkpoint path or true.
        "resume_from_checkpoint": None,

        # Leave false unless you have verified compatibility with your TRL version.
        "use_liger_kernel": False,

        # Newer TRL uses max_length; older Unsloth examples use max_seq_length.
        # The script maps model.max_seq_length to whichever argument exists.
        #"truncation_mode": "keep_start",

        # Dataloader workers for streaming. Increase only after verifying throughput.
        "dataloader_num_workers": 0,
    },

    "lora": {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_rslora": False,
        "loftq_config": None,
        "modules_to_save": None,
    },

    "save": {
        "save_final": True,
        "final_dir_name": "final",

        "save_merged_16bit": False,
        "merged_16bit_dir_name": "merged_16bit",

        "save_gguf": False,
        "gguf_dir_name": "gguf",
        "gguf_quantization_method": "q8_0",
    },
}


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


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_dot_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Bad override {override!r}; expected key.path=value.")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.split(".")
        target = cfg
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = parse_scalar(raw_value)
    return cfg


def load_config(path: str, overrides: Iterable[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    cfg = apply_dot_overrides(cfg, overrides)
    return normalize_config(cfg)


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    method = str(cfg["training"]["method"]).strip().lower()
    if method not in {"lora", "full"}:
        raise ValueError("training.method must be 'lora' or 'full'.")
    cfg["training"]["method"] = method

    if cfg["clearml"]["task_name"] is None:
        cfg["clearml"]["task_name"] = cfg["project"]["run_name"]

    if method == "full":
        cfg["model"]["full_finetuning"] = True
        cfg["model"]["load_in_4bit"] = False
        cfg["model"]["load_in_8bit"] = False
        cfg["model"]["load_in_16bit"] = False
    else:
        cfg["model"]["full_finetuning"] = False
        if not any(
            bool(cfg["model"].get(flag))
            for flag in ("load_in_4bit", "load_in_8bit", "load_in_16bit")
        ):
            cfg["model"]["load_in_16bit"] = True

    true_modes = [
        name
        for name in ("load_in_4bit", "load_in_8bit", "load_in_16bit", "full_finetuning")
        if bool(cfg["model"].get(name))
    ]
    if len(true_modes) > 1:
        raise ValueError(f"Only one load/training mode may be true. Got: {true_modes}")

    if cfg["data"].get("streaming", True):
        max_steps = int(cfg["training"].get("max_steps", -1))
        if max_steps <= 0:
            raise ValueError(
                "With data.streaming=true, set training.max_steps > 0. "
                "Streaming IterableDataset has no reliable __len__."
            )

    if not cfg["data"].get("validation_jsonl"):
        cfg["training"]["eval_strategy"] = "no"

    return cfg


def set_environment(cfg: Dict[str, Any]) -> None:
    for key, value in (cfg.get("env") or {}).items():
        if value is not None:
            os.environ[str(key)] = str(value)


def filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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


def torch_dtype_from_string(torch_module: Any, dtype_name: Optional[str]) -> Optional[Any]:
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


def is_moe_like_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(token in lowered for token in ("a3b", "a4b", "a10b", "a17b", "moe"))


def choose_unsloth_loader(cfg: Dict[str, Any], fast_language_model: Any, fast_model: Any) -> Any:
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


def init_clearml(cfg: Dict[str, Any], config_path: str) -> Optional[Any]:
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


def configure_tokenizer(tokenizer: Any, cfg: Dict[str, Any]) -> None:
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
        except Exception:
            LOG.warning("Could not auto-set Qwen EOS token; using tokenizer default.")

    if pad_setting and pad_setting != "auto":
        tokenizer.pad_token = pad_setting
    elif pad_setting == "auto" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOG.info("Set pad_token to eos_token.")


def validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")

    allowed_roles = {"system", "user", "assistant", "tool"}

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {idx} is not an object.")
        role = message.get("role")
        if role not in allowed_roles:
            raise ValueError(f"message {idx} has unsupported role: {role!r}")
        if "content" not in message:
            raise ValueError(f"message {idx} is missing content.")

    if messages[-1].get("role") != "assistant":
        raise ValueError("last message must be an assistant message for SFT.")


def normalize_prompt_completion_example(
    example: Dict[str, Any],
    messages_field: str,
    do_validate: bool,
) -> Dict[str, Any]:
    messages = example[messages_field]

    if do_validate:
        validate_messages(messages)

    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message must be assistant.")

    prompt = messages[:-1]
    completion = [messages[-1]]

    return {
        "prompt": prompt,
        "completion": completion,
    }


def maybe_take(dataset: Any, n: Optional[int], streaming: bool) -> Any:
    if n is None:
        return dataset
    n = int(n)
    if n <= 0:
        return dataset.take(0) if streaming else dataset.select(range(0))
    if streaming:
        return dataset.take(n)
    return dataset.select(range(min(n, len(dataset))))


def load_one_split(cfg: Dict[str, Any], split_name: str, data_files: Union[str, list[str]]) -> Any:
    from datasets import load_dataset

    streaming = bool(cfg["data"].get("streaming", True))
    messages_field = cfg["data"].get("messages_field", "messages")
    do_validate = bool(cfg["data"].get("validate_messages", True))

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

    dataset = dataset.map(
        lambda ex: normalize_prompt_completion_example(ex, messages_field, do_validate),
    )

    return dataset


def load_streaming_datasets(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_dataset = load_one_split(cfg, "train", cfg["data"]["train_jsonl"])

    eval_dataset = None
    validation_jsonl = cfg["data"].get("validation_jsonl")
    if validation_jsonl and str(cfg["training"].get("eval_strategy", "no")).lower() != "no":
        eval_dataset = load_one_split(cfg, "validation", validation_jsonl)

    return {
        "train": train_dataset,
        "validation": eval_dataset,
    }


def load_model_and_tokenizer(cfg: Dict[str, Any]) -> tuple[Any, Any, Any]:
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

    configure_tokenizer(tokenizer, cfg)

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

    return model, tokenizer, loader_cls


def count_parameters(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


def make_sft_config(cfg: Dict[str, Any], tokenizer: Any, SFTConfig: Any) -> Any:
    training_cfg = copy.deepcopy(cfg["training"])

    resume_from_checkpoint = training_cfg.pop("resume_from_checkpoint", None)
    if resume_from_checkpoint is not None:
        LOG.info("resume_from_checkpoint is used in trainer.train(), not SFTConfig.")

    method = training_cfg.pop("method", None)
    if method is not None:
        pass

    max_seq_length = int(cfg["model"]["max_seq_length"])

    # Ensure assistant-only masking is always active.
    training_cfg["assistant_only_loss"] = True

    # Conversational datasets should not use dataset_text_field="text".
    training_cfg.pop("dataset_text_field", None)

    # Set EOS in SFTConfig when supported.
    if getattr(tokenizer, "eos_token", None):
        training_cfg["eos_token"] = tokenizer.eos_token

    # TRL version compatibility: newer TRL uses max_length, older Unsloth examples use max_seq_length.
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sft_params:
        training_cfg["max_length"] = max_seq_length
    elif "max_seq_length" in sft_params:
        training_cfg["max_seq_length"] = max_seq_length

    # eval_strategy/evaluation_strategy compatibility.
    if "eval_strategy" in training_cfg and "eval_strategy" not in sft_params and "evaluation_strategy" in sft_params:
        training_cfg["evaluation_strategy"] = training_cfg.pop("eval_strategy")
    elif "evaluation_strategy" in training_cfg and "evaluation_strategy" not in sft_params and "eval_strategy" in sft_params:
        training_cfg["eval_strategy"] = training_cfg.pop("evaluation_strategy")

    # Some TRL versions use report_to="none"; some accept [].
    if training_cfg.get("report_to") is None:
        training_cfg["report_to"] = []

    filtered = filter_kwargs(SFTConfig.__init__, training_cfg)
    dropped = sorted(set(training_cfg) - set(filtered))
    if dropped:
        LOG.info("Dropping SFTConfig args unsupported by installed TRL: %s", dropped)

    return SFTConfig(**filtered)


def make_clearml_callback(task: Any):
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

    return trainer


def inspect_first_batch(trainer: Any, tokenizer: Any) -> None:
    import torch

    LOG.info("Inspecting first train dataloader batch.")
    batch = next(iter(trainer.get_train_dataloader()))

    labels = batch.get("labels")
    input_ids = batch.get("input_ids")

    if labels is None:
        raise RuntimeError("Batch has no labels. SFTTrainer did not prepare labels.")

    labels_tensor = labels if torch.is_tensor(labels) else torch.tensor(labels)
    supervised = labels_tensor.ne(-100).sum().item()
    total = labels_tensor.numel()
    pct = 100.0 * supervised / total if total else 0.0

    LOG.info("Supervised labels: %d / %d (%.4f%%)", supervised, total, pct)

    if supervised == 0:
        raise RuntimeError(
            "All labels are -100. assistant_only_loss masking failed. "
            "Check that the model chat template has {% generation %} markers or set training.chat_template_path."
        )

    if input_ids is not None:
        input_tensor = input_ids if torch.is_tensor(input_ids) else torch.tensor(input_ids)
        first_ids = input_tensor[0].detach().cpu().tolist()
        label_ids = labels_tensor[0].detach().cpu().tolist()

        decoded_input = tokenizer.decode(first_ids[:512], skip_special_tokens=False)

        supervised_token_ids = [
            token_id
            for token_id, label_id in zip(first_ids, label_ids)
            if label_id != -100
        ]
        decoded_supervised = tokenizer.decode(supervised_token_ids[:256], skip_special_tokens=False)

        LOG.info("First example rendered prefix:\n%s", decoded_input)
        LOG.info("First example supervised-token prefix:\n%s", decoded_supervised)


def save_outputs(cfg: Dict[str, Any], model: Any, tokenizer: Any, trainer: Any, clearml_task: Optional[Any]) -> None:
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