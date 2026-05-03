# Configuration Reference

All configuration is done via YAML files.  The system uses [OmegaConf](https://omegaconf.readthedocs.io/) for loading and merging.

## Inheritance

Any config file may declare `_base_: <relative_path>` to inherit from another config file.  The current file's keys are merged on top of the base.  Inheritance is recursive.

```yaml
_base_: base.yaml

model:
  name_or_path: "ufal/robeczech-base"
```

## CLI Overrides

Dot-notation key=value pairs passed after the config path override any config value:

```bash
python train.py configs/base.yaml optimizer.lr=3e-5 training.batch_size=8
```

---

## Top-level Keys

### `tasks`

Defines classification tasks.  Each key is a task name; each value has a `classes` list.

```yaml
tasks:
  topic:
    classes: [politics, economy, culture, sport, science, other]
  document_type:
    classes: [article, letter, decree, advertisement, notice]
```

Ground truth for each task is stored under the same key in the JSONL data rows.

---

### `model`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name_or_path` | str | — | HuggingFace model ID or local path |
| `dropout` | float | `0.1` | Dropout before classification heads |
| `pooling` | str | `"cls"` | `"cls"` (first token) or `"mean"` (masked mean) |
| `freeze_encoder_layers` | int | `0` | Number of initial encoder layers to freeze |
| `compile` | bool | `false` | Wrap model with `torch.compile` for faster training (PyTorch ≥ 2.0; adds ~1 min warm-up on first step) |

---

### `data`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `train` | str | — | Path to training JSONL |
| `val` | list or str | — | Validation JSONL paths; each entry has `path` and `name` |
| `max_length` | int | `512` | Tokenizer max sequence length |
| `num_workers` | int | `4` | DataLoader worker processes |

`val` supports multiple datasets for separate evaluation:

```yaml
data:
  val:
    - path: "data/val_in_domain.jsonl"
      name: "val_in"
    - path: "data/val_out_domain.jsonl"
      name: "val_out"
```

---

### `training`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `batch_size` | int | `16` | Training batch size per GPU |
| `val_batch_size` | int | `32` | Validation batch size |
| `max_steps` | int | — | Total optimizer update steps |
| `val_every_n_steps` | int | `500` | Run validation every N steps |
| `grad_accumulation` | int | `1` | Gradient accumulation steps |
| `gradient_clip_val` | float | `1.0` | Gradient norm/value clip (`null` to disable) |
| `gradient_clip_algorithm` | str | `"norm"` | `"norm"` or `"value"` |
| `precision` | str | `"bf16-mixed"` | PyTorch Lightning precision string |
| `devices` | str/int | `"auto"` | Number of GPUs or `"auto"` |
| `strategy` | str | `"auto"` | `"auto"`, `"ddp"`, `"deepspeed"`, etc. |
| `log_every_n_steps` | int | `10` | Logging frequency |

---

### `optimizer`

AdamW is used by default.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lr` | float | — | Learning rate |
| `weight_decay` | float | `0.01` | L2 regularization (not applied to bias/LN) |
| `betas` | list | `[0.9, 0.999]` | Adam β₁, β₂ |
| `eps` | float | `1e-8` | Adam ε |

---

### `scheduler`

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Scheduler type (see below) |
| `interval` | str | `"step"` or `"epoch"` |
| `frequency` | int | How often the scheduler steps |

#### Scheduler types

| Name | Extra keys | Description |
|------|-----------|-------------|
| `linear_warmup_cosine` | `warmup_steps`, `total_steps` | Warmup then cosine decay (HuggingFace) |
| `linear_warmup` | `warmup_steps`, `total_steps` | Warmup then linear decay |
| `constant_warmup` | `warmup_steps` | Warmup then constant LR |
| `cosine` | `T_max`, `eta_min` | PyTorch CosineAnnealingLR |
| `reduce_on_plateau` | `mode`, `factor`, `patience` | ReduceLROnPlateau |
| `one_cycle` | `total_steps`, `max_lr`, `pct_start` | OneCycleLR |

---

### `checkpoint`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable checkpoint saving |
| `dirpath` | str | `"checkpoints"` | Directory for checkpoint files |
| `save_top_k` | int | `3` | Keep best N checkpoints |
| `save_last` | bool | `true` | Always save the last checkpoint |
| `mode` | str | `"min"` | `"min"` or `"max"` for the monitor metric |
| `monitor` | str | first val loss | Metric to monitor for best-k selection |

---

### `clearml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Enable ClearML logging |
| `project` | str | `"text_classification"` | ClearML project name |
| `task_name` | str | `"training"` | ClearML task name |

---

### Other

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `log_level` | str | `"INFO"` | Python logging level |
| `seed` | int | `42` | Random seed for reproducibility |
| `deterministic` | bool | `false` | Force deterministic CUDA ops |
