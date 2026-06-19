import yaml
import copy
from typing import Any, Iterable, Mapping

DEFAULT_CONFIG: dict[str, Any] = {
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
    "logging": {
        "log_token_counts": True,
        "token_count_log_every": 10,
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


def parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def apply_dot_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dot-separated key overrides to a config dictionary. For example, --set training.max_steps=20000"""
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


def load_config(path: str, overrides: Iterable[str]) -> dict[str, Any]:
    """ Load YAML config and apply dot-separated overrides. Returns a normalized config dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    cfg = apply_dot_overrides(cfg, overrides)
    return normalize_config(cfg)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply config defaults and validate settings. This is a good place to catch incompatible or missing settings early."""
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