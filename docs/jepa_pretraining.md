# JEPA Pretraining Reference

Byte-segment JEPA (Joint Embedding Predictive Architecture) pretraining for tokenizer-free document representations.  The approach encodes raw UTF-8 bytes in 32-byte canonical segments, trains a student encoder to predict teacher embeddings of masked/corrupted regions, and produces segment-level and document-level representations.

---

## Quick Start

```bash
# Install (already included in the package)
pip install -e "."

# Run the full smoketest (tiny model, 10 steps, CPU)
python jepa_pretrain.py configs/jepa_smoketest.yaml

# Train with the base configuration
python jepa_pretrain.py configs/jepa_base.yaml \
    data.train=/path/to/train.jsonl \
    data.val=/path/to/val.jsonl

# Override individual keys on the CLI
python jepa_pretrain.py configs/jepa_base.yaml \
    model.seg_dim=256 \
    training.batch_size=16 \
    optimizer.lr=3e-4
```

---

## Data Format

Training data must be in **JSONL** format.  Each line is a JSON object with two required fields:

| Field | Type | Description |
|-------|------|-------------|
| `id`   | str  | Unique document identifier |
| `text` | str  | Raw text; any Unicode, arbitrary length |

Example:

```jsonl
{"id": "doc-001", "text": "Lorem ipsum dolor sit amet..."}
{"id": "doc-002", "text": "Another document in raw form."}
```

The preprocessor converts text to UTF-8 bytes and splits them into 32-byte canonical segments.  Very long documents are truncated at `max_segments` (default 2048).

---

## Configuration Reference

Config files use [OmegaConf](https://omegaconf.readthedocs.io/) YAML with `_base_` inheritance (same as the classification configs; see `docs/configuration.md`).

### `model`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `byte_dim` | int | `256` | Byte embedding and local CNN dimension |
| `seg_dim` | int | `512` | Segment-level transformer dimension |
| `pred_dim` | int | `512` | Predictor output / teacher target dimension |
| `n_byte_blocks` | int | `4` | Number of local ByteConv residual blocks |
| `n_encoder_layers` | int | `12` | Transformer encoder layers |
| `n_heads` | int | `8` | Attention heads |
| `ffn_dim` | int | `2048` | FFN hidden dimension (SwiGLU) |
| `n_predictor_layers` | int | `4` | Predictor transformer layers |
| `max_segments` | int | `2048` | Maximum segments per document |
| `kernel_size` | int | `5` | Byte-level Conv1D kernel size |
| `dropout` | float | `0.1` | Dropout probability |
| `byte_dropout` | float | `0.05` | Independent byte-embedding dropout |
| `ema_momentum` | float | `0.996` | Starting EMA teacher momentum |
| `teacher_target_layers` | list\|null | `null` | Encoder layers averaged for teacher targets; `null` = last layer only |

### `ema`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `momentum_start` | float | `0.996` | EMA momentum at step 0 |
| `momentum_end` | float | `0.9999` | EMA momentum at final step (cosine schedule) |

### `loss`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `segment` | float | `1.0` | Weight on segment JEPA cosine loss |
| `document` | float | `0.2` | Weight on document consistency loss |
| `variance` | float | `0.05` | Weight on variance regularisation (VICReg) |
| `covariance` | float | `0.01` | Weight on covariance regularisation (VICReg) |

### `data`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `train` | str | — | Path to training JSONL (**required**) |
| `val` | str\|list | — | Validation JSONL path(s); same multi-dataset format as classifier |
| `max_segments` | int | `2048` | Maximum segments per document |
| `num_workers` | int | `4` | DataLoader workers |
| `pin_memory` | bool | `true` | Pin DataLoader memory for GPU transfers |

### `corruption`

Controls how the student view is generated from clean segments.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `missing_span_prob` | float | `0.15` | Probability a contiguous span is dropped |
| `missing_span_frac_min` | float | `0.1` | Minimum fraction of document in a missing span |
| `missing_span_frac_max` | float | `0.4` | Maximum fraction of document in a missing span |
| `segment_dropout_prob` | float | `0.05` | Per-segment dropout probability |
| `foreign_replace_frac` | float | `0.05` | Fraction of segments replaced with random foreign bytes |
| `local_reorder_prob` | float | `0.1` | Probability of reordering a local block of segments |
| `truncate_prob` | float | `0.2` | Probability of randomly truncating the student view |
| `truncate_frac_max` | float | `0.3` | Maximum fraction to truncate |
| `byte_corrupt_prob` | float | `0.02` | Per-byte corruption probability inside retained segments |
| `byte_heavy_frac` | float | `0.1` | Fraction of segments that receive heavy byte noise |

### `training`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `batch_size` | int | `32` | Documents per step (**required**) |
| `max_steps` | int | `100000` | Total training steps (**required**) |
| `val_every_n_steps` | int | `2000` | Validate every N steps |
| `precision` | str | `"bf16-mixed"` | Trainer precision; `"32"` for debugging |
| `devices` | int | `1` | Number of GPUs |
| `strategy` | str | `"auto"` | Lightning distributed strategy |
| `gradient_clip_val` | float | `1.0` | Max gradient norm |
| `accumulate_grad_batches` | int | `1` | Gradient accumulation steps |

### `optimizer`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lr` | float | `1e-4` | Peak learning rate (**required**) |
| `weight_decay` | float | `0.01` | AdamW weight decay |
| `betas` | list | `[0.9, 0.999]` | AdamW beta coefficients |
| `eps` | float | `1e-8` | AdamW epsilon |

### `scheduler`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | str | `"linear_warmup_cosine"` | Schedule type; also `"constant_warmup"`, `"cosine"`, `"constant"` |
| `warmup_steps` | int | `3000` | Linear warmup steps |
| `min_lr_ratio` | float | `0.1` | Minimum LR as fraction of peak (cosine floor) |

### `checkpoint`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Save checkpoints |
| `dirpath` | str | `"checkpoints"` | Checkpoint directory |
| `monitor` | str | first val loss | Metric to monitor |
| `mode` | str | `"min"` | `"min"` or `"max"` |
| `save_top_k` | int | `3` | Checkpoints to keep |
| `save_last` | bool | `true` | Always save the last checkpoint |

---

## Architecture Overview

```
                  ┌─────────────────────────────────┐
raw text  ──────► │  text_to_canonical_segments()   │  N × 32 bytes
                  └─────────────────────────────────┘
                              │
              ┌───────────────┴──────────────┐
        (clean view)                  (student view)
              │                             │  corruption, dropout,
              │                             │  missing spans, reorder
              ▼                             ▼
       ByteSegmentEncoder           ByteSegmentEncoder (student)
       (EMA teacher, frozen)        (learnable)
              │                             │
              ▼                             ▼
    teacher_seg_targets            predicted_segments
    teacher_doc_target             predicted_doc
              │                             │
              └──────────── loss ───────────┘
                  segment JEPA (cosine)
                  document consistency
                  VICReg variance / covariance
```

### Key Components

| Component | Description |
|-----------|-------------|
| `ByteInputEmbedding` | Byte value (vocab 259) + offset (32) + corruption type (7) → LayerNorm |
| `LocalByteProcessor` | Stacked `ByteConvBlock` residuals (Conv1D + SwiGLU); works on 32-byte windows |
| `ByteToSegmentReducer` | Aggregates 32 byte embeddings to one segment vector: mean + attn + first + last → linear |
| `TransformerEncoderWithIntermediates` | Pre-LN transformer; returns all layer outputs for intermediate target averaging |
| `SegmentPredictor` | Predicts teacher targets from student context via cross-attention; output L2-normalised |
| `ByteSegmentJEPA` | Combines student encoder, EMA teacher, predictor; exposes `update_teacher(momentum)` |

### Vocabulary

The byte vocabulary size is **259**:
- 0–255: raw byte values
- 256: `PAD_BYTE` — padding to 32-byte segment boundary
- 257: `MASK_BYTE` — masked / missing-span indicator
- 258: `NOISE_BYTE` — foreign / random noise bytes

---

## Corruption Types

Each segment in the student view is tagged with a `CorruptionType` label used as an additional embedding signal:

| Type | Value | Description |
|------|-------|-------------|
| `CLEAN` | 0 | Unmodified segment |
| `PARTIALLY_CORRUPT` | 1 | A few bytes replaced with noise |
| `HEAVILY_CORRUPT` | 2 | Many bytes replaced with noise |
| `MASKED` | 3 | Entire segment replaced with MASK_BYTE |
| `FOREIGN` | 4 | Segment replaced with random foreign bytes |
| `REORDERED` | 5 | Segment position permuted within a local block |
| `PADDING` | 6 | Zero-padded segment beyond document end |

---

## Curriculum Training

Training progresses through 5 stages, advancing automatically when validation loss plateaus:

| Stage | Min chars | Max chars | Notes |
|-------|-----------|-----------|-------|
| `sanity` | 500 | 1 000 | Very short documents |
| `short` | 500 | 2 000 | Short documents |
| `medium` | 1 000 | 5 000 | Medium-length documents |
| `long` | 3 000 | 10 000 | Long documents |
| `full` | 500 | 10 000 | All document lengths |

Curriculum is driven by `CurriculumScheduler(patience=3)`.  The `JEPAPretrainingModule` calls `report_val_loss` at each validation epoch; when loss stops improving for `patience` checks the scheduler advances to the next stage and the data module filters documents by the new character range.

---

## Monitoring

The following metrics are logged at each validation step:

| Metric | Description |
|--------|-------------|
| `val/<name>/loss` | Total weighted loss |
| `val/<name>/loss/segment` | Segment JEPA cosine loss |
| `val/<name>/loss/document` | Document consistency loss |
| `val/<name>/loss/variance` | Variance regularisation |
| `val/<name>/loss/covariance` | Covariance regularisation |
| `val/<name>/cosine_seg` | Mean cosine similarity, student vs teacher segments |
| `val/<name>/cosine_doc` | Cosine similarity, student vs teacher document |
| `val/<name>/embedding_std` | Mean per-dim std of teacher segment targets (collapse monitor; should stay > 0.1) |

---

## Example Config

```yaml
# configs/my_run.yaml
_base_: jepa_base.yaml

data:
  train: /data/czech_news/train.jsonl
  val:
    - path: /data/czech_news/val.jsonl
      name: news

training:
  batch_size: 64
  max_steps: 200000
  precision: "bf16-mixed"
  devices: 4

optimizer:
  lr: 2e-4
```

```bash
python jepa_pretrain.py configs/my_run.yaml
```

---

## Extracting Embeddings

After pretraining, load the student encoder directly:

```python
import torch
from omegaconf import OmegaConf
from text_classification.jepa.model import ByteSegmentJEPA
from text_classification.jepa.corruption import text_to_canonical_segments, CorruptionType, PAD_BYTE, BYTE_VOCAB_SIZE, SEGMENT_SIZE

# Load checkpoint
ckpt = torch.load("checkpoints/last.ckpt", map_location="cpu")
cfg = OmegaConf.load("configs/jepa_base.yaml")

model = ByteSegmentJEPA(**cfg.model)
# strip "model." prefix from Lightning state dict
state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
model.load_state_dict(state)
model.eval()

# Encode a document
text = "Prague is the capital of the Czech Republic."
segments = text_to_canonical_segments(text)
N = len(segments)

clean_bytes = torch.zeros(1, N, SEGMENT_SIZE, dtype=torch.long)
clean_types = torch.zeros(1, N, SEGMENT_SIZE, dtype=torch.long)  # all CLEAN
for i, seg in enumerate(segments):
    for j, b in enumerate(seg):
        clean_bytes[0, i, j] = b

canonical_positions = torch.arange(N).unsqueeze(0)  # (1, N)
canonical_mask = torch.ones(1, N, dtype=torch.bool)  # True = valid

with torch.no_grad():
    out = model(
        clean_byte_values=clean_bytes,
        clean_byte_types=clean_types,
        canonical_positions=canonical_positions,
        canonical_mask=canonical_mask,
        student_bytes=clean_bytes,  # no corruption for inference
        student_byte_types=clean_types,
        student_positions=canonical_positions,
        student_mask=canonical_mask,
    )

# out["teacher_doc_targets"] shape: (1, pred_dim) — document embedding
# out["teacher_seg_targets"] shape: (1, N, pred_dim) — per-segment embeddings
doc_embedding = out["teacher_doc_targets"]  # use teacher for stable embeddings
```
