# LLM Text Classification

Fine-tune causal language models for multi-task multi-label text classification on the same datasets and tasks as the BERT-based pipeline.

Rather than classification heads, the LLM is trained to produce a **structured JSON object** with one entry per configured task:

```json
{
  "communicative_mode": {
    "reason": "The text narrates events in sequence.",
    "classes": ["narration"]
  },
  "structural_form": {
    "reason": "Continuous paragraphs with no list structure.",
    "classes": ["continuous_prose"]
  }
}
```

---

## Table of Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Configuration reference](#configuration-reference)
4. [Prompt and output format](#prompt-and-output-format)
5. [Training](#training)
   - [Single GPU](#single-gpu)
   - [Multi-GPU (DDP)](#multi-gpu-ddp)
6. [Evaluation](#evaluation)
7. [Metrics](#metrics)
8. [Code structure](#code-structure)

---

## Installation

```bash
# Core LLM dependencies (HuggingFace backend + LoRA + metrics)
pip install -e ".[llm]"

# Optional: unsloth backend (faster single-GPU training)
pip install -e ".[unsloth]"
```

---

## Quick start

```bash
# Edit configs/llm_base.yaml to point at your data and model
python llm_train.py configs/llm_base.yaml

# Evaluate a checkpoint
python llm_eval.py \
    --config configs/llm_base.yaml \
    --checkpoint checkpoints/last.ckpt \
    --dataset data/test.jsonl \
    --output predictions.jsonl
```

---

## Configuration reference

`configs/llm_base.yaml` is the canonical template. All sections below are top-level keys.

### `tasks`

Same format as the BERT config. Only the tasks listed here will appear in the LLM's output JSON (i.e. the model is prompted to classify only those tasks).

```yaml
tasks:
  communicative_mode:
    classes: [narration, description, ...]
```

### `model`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name_or_path` | str | — | HuggingFace model identifier or local path |
| `backend` | str | `"huggingface"` | `"huggingface"` or `"unsloth"` |
| `quantization` | str\|null | `null` | `null`, `"4bit"`, or `"8bit"` |
| `max_seq_length` | int | `4096` | Used by the unsloth backend only |
| `lora.enabled` | bool | `true` | Enable LoRA / QLoRA |
| `lora.r` | int | `16` | LoRA rank |
| `lora.alpha` | int | `32` | LoRA scaling factor |
| `lora.dropout` | float | `0.05` | LoRA dropout |
| `lora.target_modules` | list[str] | `["q_proj","k_proj","v_proj","o_proj"]` | Modules to apply LoRA to |

### `llm`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_reason` | bool | `false` | Ask the model to produce a `reason` field per task |
| `max_input_length` | int | `2048` | Maximum tokens for the prompt |
| `max_output_length` | int | `512` | Maximum new tokens during generation |
| `prompt_template` | str\|null | `null` | Custom prompt template string; `null` uses the built-in default |

### `generation`

Controls autoregressive sampling during validation and evaluation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `do_sample` | bool | `false` | Use sampling instead of greedy decoding |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_p` | float | `1.0` | Nucleus sampling probability |

### `data`

Same format as the BERT config: `data.train` (path), `data.val` (list of `{path, name}` records).

### `training`

Same keys as the BERT config: `batch_size`, `val_batch_size`, `max_steps`, `val_every_n_steps`, `grad_accumulation`, `gradient_clip_val`, `gradient_clip_algorithm`, `precision`, `devices`, `strategy`, `log_every_n_steps`.

### `optimizer`

`lr`, `weight_decay`, `betas`, `eps` — same as BERT config.

### `scheduler`

`name`, `warmup_steps`, `total_steps` — same schedulers as BERT config: `linear_warmup`, `linear_warmup_cosine`, `cosine`, `constant_warmup`, `reduce_on_plateau`, `one_cycle`.

### `validation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_samples` | int\|null | `null` | Cap the number of examples used per validation dataset (useful for fast feedback) |
| `bertscore_model` | str\|null | `null` | HuggingFace model for BERTScore; `null` disables it |

### `clearml`

`enabled`, `project`, `task_name` — same as BERT config.

---

## Prompt and output format

### Default prompt

```
You are a document classifier. Analyse the text and output ONLY a valid JSON
object — no explanation, no markdown fences.

For each of the following tasks, choose one or more of the listed classes that
best describe the text. For each task also provide a short reasoning string in
the "reason" key (output it before the "classes" key).

Tasks and allowed classes:
  communicative_mode: ["narration", "description", ...]
  structural_form: ["continuous_prose", ...]

Text:
<document text here>

Output:
```

The model should respond with a JSON object containing exactly the listed tasks as keys.

### Custom prompt template

Set `llm.prompt_template` in the config to override the default template. The string must contain three placeholders:

- `{reason_instruction}` — replaced with the reason instruction (or empty string when `use_reason=false`)
- `{task_list}` — replaced with the formatted task/class list
- `{text}` — replaced with the document text

### Chat-template models

When the tokenizer exposes a `chat_template` (Llama-3, Mistral-Instruct, etc.), the dataset automatically calls `tokenizer.apply_chat_template()` to wrap the prompt as a user message. The target JSON is appended as the assistant response. Set the tokenizer's chat template to `null` in the config if you prefer raw concatenation.

---

## Training

### Single GPU

```bash
python llm_train.py configs/llm_base.yaml
# or using the installed entry point:
tc-llm-train configs/llm_base.yaml
```

Override any config key via dot-notation:

```bash
python llm_train.py configs/llm_base.yaml \
    optimizer.lr=1e-4 \
    training.batch_size=2 \
    training.grad_accumulation=8
```

### Multi-GPU (DDP)

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 llm_train.py configs/llm_base.yaml training.strategy=ddp

# Using Accelerate
accelerate launch llm_train.py configs/llm_base.yaml
```

Set `training.strategy: ddp` (or `fsdp`) in the config for multi-GPU runs. The unsloth backend supports multi-GPU via Accelerate/DDP in the same way.

---

## Evaluation

```bash
python llm_eval.py \
    --config configs/llm_base.yaml \
    --checkpoint checkpoints/last.ckpt \
    --dataset data/test.jsonl \
    --output predictions.jsonl \
    --batch-size 8
```

The output JSONL mirrors the input with two additional fields per row:
- `_llm_pred` — the parsed prediction dict (or an empty dict if JSON parsing failed)
- `_llm_raw` — the raw decoded model output string

---

## Metrics

### Classification (per task)

- **mAP** — mean Average Precision across classes
- **precision/macro**, **recall/macro**, **f1/macro** — threshold-0.5 macro metrics

All metrics inherit from `MultiLabelMetrics` and are logged as `val/<dataset>/<task>/<metric>`.

### Reason similarity

- **ROUGE-L** — always computed when gold `reason` text is present; logged as `val/<dataset>/<task>/reason/rouge_l`.
- **BERTScore F1** — logged as `val/<dataset>/<task>/reason/bertscore_f1` when `validation.bertscore_model` is set.

### Overall

- `val/<dataset>/overall/mAP` — macro mean of per-task mAP (primary checkpoint monitor metric).

---

## Code structure

```
text_classification/llm/
├── __init__.py
├── train.py              ← tc-llm-train entry point
├── eval.py               ← tc-llm-eval entry point
├── eval_main.py          ← evaluation logic
├── data/
│   ├── prompt.py         ← build_prompt, build_target
│   └── dataset.py        ← LLMClassificationDataset, LLMClassificationDataModule
├── models/
│   └── llm_model.py      ← load_model_and_tokenizer (hf + unsloth backends)
├── training/
│   ├── lightning_module.py  ← LLMClassificationModule
│   └── train_main.py     ← training script main()
└── metrics/
    └── generation.py     ← parse_json_output, compute_rouge_l, GenerationMetrics

text_classification/utils/
└── optimizers.py         ← build_optimizer, build_scheduler (shared)

llm_train.py              ← convenience top-level script
llm_eval.py               ← convenience top-level script
configs/llm_base.yaml     ← template configuration
tests/llm/
├── test_dataset.py
├── test_metrics.py
└── test_model.py
```
