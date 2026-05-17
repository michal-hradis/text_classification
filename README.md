# Text Classification

Multi-label text classification framework for Czech OCR documents, built on PyTorch Lightning and HuggingFace Transformers.  Supports multiple tasks per example, missing ground-truth masking, multi-GPU training, and experiment tracking via ClearML.

---

## Project Structure

```
text_classification/
├── configs/                   # Example YAML configurations
│   ├── base.yaml              # Base config (inherited by all others)
│   ├── robeczech_base.yaml    # ufal/robeczech-base
│   ├── xlm_roberta_base.yaml  # FacebookAI/xlm-roberta-base
│   ├── xlm_roberta_large.yaml # FacebookAI/xlm-roberta-large
│   ├── hplt_bert_czech.yaml   # HPLT/hplt_bert_base_2_0_ces-Latn
│   └── mdeberta_base.yaml     # microsoft/mdeberta-v3-base
├── text_classification/
│   ├── data/
│   │   └── dataset.py         # Dataset, DataModule, collate_fn
│   ├── models/
│   │   └── classifier.py      # TransformerClassifier, ClassificationHead
│   ├── metrics/
│   │   └── multilabel.py      # MultiLabelMetrics (mAP, AP, P, R, F1)
│   ├── training/
│   │   └── lightning_module.py # TextClassificationModule (pl.LightningModule)
│   └── utils/
│       ├── config.py          # load_config, validate_config
│       └── logging.py         # setup_logging, ClearMLLogger
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_model.py
├── train.py                   # Training entry point
├── export.py                  # Checkpoint → HuggingFace export
├── requirements.txt
└── pyproject.toml
```

---

## Fast Start

### 1. Install

```bash
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### 2. Prepare data

Each split is a **JSONL** file where every line is a JSON object:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "document": "doc-uuid",
  "text": "Dlouhý text dokumentu ...",
  "topic": {
    "classes": ["politics", "economy"],
    "reason": "The text discusses government spending."
  },
  "sentiment": null
}
```

- `text` — the document segment (500–4000 characters typical)
- `id` / `document` — UUIDs for traceability
- Ground-truth keys match task names in the config.  Set to `null` or omit the key entirely when the annotation is missing for a given task.

### 3. Define tasks

Edit a config (or start from an example) and fill in the `tasks` block:

```yaml
tasks:
  topic:
    classes: [politics, economy, culture, sport, science, other]
  document_type:
    classes: [article, letter, decree, advertisement, notice]
```

### 4. Train

```bash
python train.py configs/robeczech_base.yaml
```

With overrides:

```bash
python train.py configs/xlm_roberta_base.yaml \
    data.train=data/train.jsonl \
    optimizer.lr=3e-5 \
    training.max_steps=5000
```

### 5. Export

Save the HuggingFace encoder + classification heads from a checkpoint:

```bash
python export.py configs/robeczech_base.yaml checkpoints/last.ckpt outputs/my_model
```

The output directory will contain:
- `encoder/` — HuggingFace model + tokenizer (loadable with `AutoModel.from_pretrained`)
- `heads.pt` — classification head weights (`state_dict`)
- `config.yaml` — training config snapshot

### 6. Run tests

```bash
pytest
```

---

## Configuration Reference

See [`docs/configuration.md`](docs/configuration.md) for the full reference.

Key top-level sections:

| Section | Description |
|---------|-------------|
| `tasks` | Task names and their class lists |
| `model` | Backbone model, dropout, pooling strategy |
| `data` | Train/val JSONL paths, max length, workers |
| `training` | Batch size, max steps, gradient clipping, precision |
| `optimizer` | AdamW learning rate, weight decay, betas |
| `scheduler` | LR scheduler type and parameters |
| `checkpoint` | Checkpoint directory and selection metric |
| `clearml` | ClearML project/task name, enable flag |

Config files support **inheritance** via `_base_`:

```yaml
_base_: base.yaml

model:
  name_or_path: "ufal/robeczech-base"
optimizer:
  lr: 3.0e-5
```

CLI overrides use dot-notation and are applied last:

```bash
python train.py my_config.yaml training.batch_size=8 optimizer.lr=1e-5
```

---

## Supported Models

All models are loaded via `transformers.AutoModel`; any HuggingFace model that provides `last_hidden_state` works.

| Model | Identifier |
|-------|-----------|
| HPLT Czech BERT 2.0 | `HPLT/hplt_bert_base_2_0_ces-Latn` |
| HPLT Czech BERT | `HPLT/hplt_bert_base_cs` |
| mmBERT small | `jhu-clsp/mmBERT-small` |
| mmBERT base | `jhu-clsp/mmBERT-base` |
| RobeCzech base v1.1 | `ufal/robeczech-base` |
| Czert-B base cased | `UWB-AIR/Czert-B-base-cased` |
| Czert-A base uncased | `UWB-AIR/Czert-A-base-uncased` |
| Small-E-Czech | `Seznam/small-e-czech` |
| mDeBERTa v3 base | `microsoft/mdeberta-v3-base` |
| XLM-RoBERTa base | `FacebookAI/xlm-roberta-base` |
| XLM-RoBERTa large | `FacebookAI/xlm-roberta-large` |
| XLM-RoBERTa XL | `facebook/xlm-roberta-xl` |
| XLM-RoBERTa XXL | `facebook/xlm-roberta-xxl` |
| SlavicBERT | `DeepPavlov/bert-base-bg-cs-pl-ru-cased` |
| mBERT | `bert-base-multilingual-cased` |

Custom PyTorch encoders can be injected via `TransformerClassifier.from_custom_encoder()`.

---

## Metrics

Computed per validation run per dataset per task:

| Metric | Description |
|--------|-------------|
| `mAP` | Mean Average Precision across all classes |
| `ap/<class>` | Average Precision for a single class |
| `precision/<class>` | Precision at threshold 0.5 |
| `recall/<class>` | Recall at threshold 0.5 |
| `f1/<class>` | F1 at threshold 0.5 |
| `precision/macro` | Macro-averaged precision |
| `recall/macro` | Macro-averaged recall |
| `f1/macro` | Macro-averaged F1 |

Log keys follow the pattern `val/<dataset_name>/<task>/<metric>`.

---

## Multi-GPU Training

Set `training.strategy: ddp` for DistributedDataParallel and `training.devices: -1` (all GPUs) or a specific count:

```yaml
training:
  strategy: "ddp"
  devices: 4
  precision: "bf16-mixed"
  grad_accumulation: 2
```

---

## ClearML Logging

```yaml
clearml:
  enabled: true
  project: "czech-ocr-classification"
  task_name: "robeczech-topic-run1"
```

Requires `clearml` installed and credentials configured (`clearml-init`).

---

## JEPA Pretraining

Tokenizer-free **byte-segment JEPA** pretraining produces dense document and segment representations from raw text without any tokenizer or supervised labels.  It can serve as an initialisation for downstream classifiers or be used to produce embeddings directly.

### Quick Start

```bash
# Smoketest (tiny model, 10 steps, CPU — stays in the repo)
python jepa_pretrain.py configs/jepa_smoketest.yaml

# Full pretraining
python jepa_pretrain.py configs/jepa_base.yaml \
    data.train=/path/to/train.jsonl \
    data.val=/path/to/val.jsonl

# Console entry point (after pip install -e .)
tc-jepa-pretrain configs/jepa_base.yaml
```

### Overview

- **Tokenizer-free**: operates on raw UTF-8 bytes in 32-byte canonical segments.
- **Student / teacher**: EMA teacher provides stable regression targets; student encoder learns to predict masked regions.
- **Curriculum**: training automatically advances through 5 document-length stages (500–10 000 chars) as validation loss plateaus.
- **Collapse monitoring**: variance and covariance regularisation (VICReg) prevent representation collapse.

See [`docs/jepa_pretraining.md`](docs/jepa_pretraining.md) for the full configuration reference, architecture details, and embedding extraction examples.