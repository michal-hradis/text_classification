This task is to extend JEPA pretraining and classification finetuning code mainly with ModernBERT-style model and initialization.

# ModernBERT-style architecture
- pre-normalization
- RoPE in both self-attention and cross-attention - Just keep in mind that the longest realistic context is <2000. RoPE can be safely used in cross-attention in predictor as the student input and decoder query tokens should have matching positions. Make sure that the student inputs have the original positions (before masking).
- GLU/GeGLU feed-forward blocks
- most linear biases removed
- Make local/global attention ratio configurable and also the size of the local context. I want to test also variants with only global attention.
- make sure that some form of fast attention is used 
- random Megatron-style initialization - most linear / embedding weights - `W ~ Normal(mean=0, std=0.02)`, # biases `b = 0`,  # residual-branch output projections, depending on implementation `W_out ~ Normal(mean=0, std=0.02 / sqrt(2 * num_layers))` --- Megatron Core’s config documents this directly: the default init_method is a zero-mean normal with std = init_method_std, usually 0.02; the default output_layer_init_method uses std / sqrt(2 * num_layers) for attention and MLP output layers; biases are zeroed.

# What is not needed
- padding, and sequence packing - the processed texts are generally of similar lenght and are able to fill the whole training context. Padding may not be a significant issue.

# Other enhancements
- Special static `doc` token is now supported. Add option to add additional static tokens - configuration should have `additional_tokens`. All these static token embeddings should be predicted by the `predictor` and the query embeddings should be learned. 

# Capability to extend existing model
- Add capability to load smaller encoder model checkpoint in to a model with more encoder layers. This should support "gradual" model training - e.g. first train 8-layer encoder, extend it to 12-layers, to 16-layers, ... The new layers should be initialized randomly and the rest of the encoder and predictor should be loaded from the checkpoint. 
- The teacher should have two options: 1) keep the original teacher encoder and freeze it (disable running average updates), 2) make a fresh copy of the student model.

---

# Implementation — Summary of Changes (2026-05-18)

## What was implemented

### Core model (`text_classification/jepa/model.py`) — complete rewrite

| Component | Details |
|-----------|---------|
| `RotaryEmbedding` | Precomputed cos/sin buffers; shared instance between encoder and predictor |
| `GLU` (+ `SwiGLU` alias) | SwiGLU or GeGLU selected via `activation=` config; residual output projection tagged `_is_residual_output=True` for scaled init |
| `_SelfAttention` | No biases; RoPE applied to segment tokens only (static tokens at index `<n_static` skip RoPE); SDPA (`scaled_dot_product_attention`) |
| `_CrossAttention` | No biases; RoPE on the segment-position portion of Q and K |
| `_make_local_attn_mask` | Sliding-window boolean mask; static token rows/cols always attend globally |
| `_TransformerLayer` | Pre-LN; local mask computed on-the-fly when `is_local=True` |
| `TransformerEncoderWithIntermediates` | New params: `max_seq_len`, `local_window_size`, `global_attention_every_n`, `activation`; `forward(x, positions=None, n_static=0, key_padding_mask=None)` |
| `_PredictorLayer` | Pre-LN; self-attn + cross-attn + GLU FFN |
| `SegmentPredictor` | `static_queries` learned param `(1, n_static, d_model)`; `target_query` learned param; no positional embedding table |
| `ByteSegmentEncoder` | No absolute positional embedding; `doc_embed` + optional `additional_token_embeds`; `_max_segments` buffer (replaces old `pos_embed.weight` for checkpoint introspection); `n_static = 1 + n_additional_tokens` |
| `_apply_megatron_init` | Embedding `N(0,0.02)`; LN weight=1/bias=0; Linear `N(0,0.02)` or scaled for residual outputs |
| `ByteSegmentJEPA` | New params: `n_additional_tokens`, `local_window_size`, `global_attention_every_n`, `activation`, `teacher_mode`, `megatron_init`; `load_partial_checkpoint()` for layer extension; `teacher_mode="frozen"` disables EMA updates |

**Backward compatibility preserved:** `forward()` returns the same four keys (`predicted_segments`, `teacher_seg_targets`, `predicted_doc`, `teacher_doc_targets`). Additional keys `predicted_additional_static` / `teacher_additional_static_targets` are only present when `n_additional_tokens > 0`.

### Local / global attention design

Layer `i` (0-based) is a **local** layer when `local_window_size > 0` AND `(i+1) % global_attention_every_n != 0`.  
Setting `local_window_size=0` gives full global attention on all layers (default, backward-compatible).

### Files changed

| File | Change |
|------|--------|
| `text_classification/jepa/model.py` | Full rewrite (~1 000 lines) |
| `text_classification/jepa/classifier.py` | New arch params; fixed `_peek_max_segments`; fixed `_freeze_encoder`; n_static-aware mean pooling |
| `text_classification/jepa/lightning_module.py` | Passes new model params; `extend_from_checkpoint` support |
| `text_classification/jepa/finetune_module.py` | Passes new arch params to `JEPAClassifier` |
| `configs/jepa_base.yaml` | Added 7 new `model:` keys with defaults |
| `configs/jepa_finetune_base.yaml` | Added 4 new `model:` keys with defaults |
| `docs/jepa_pretraining.md` | New params added to configuration reference table |
| `tests/jepa/test_lightning.py` | Removed stale reference to deleted `teacher_doc_head` attribute |

## Example config snippets

**Local/global attention (e.g. ModernBERT-style — global every 3rd layer):**
```yaml
model:
  local_window_size: 128
  global_attention_every_n: 3
```

**GeGLU instead of SwiGLU:**
```yaml
model:
  activation: "geglu"
```

**Additional static tokens:**
```yaml
model:
  n_additional_tokens: 4   # [DOC] + 4 extra → 5 static tokens total
```

**Frozen teacher (no EMA):**
```yaml
model:
  teacher_mode: "frozen"
```

**Gradual depth extension (8 → 12 layers):**
```yaml
model:
  n_encoder_layers: 12
  extend_from_checkpoint: "checkpoints/jepa_8layer/last.ckpt"
```

**Disable Megatron init (use PyTorch defaults):**
```yaml
model:
  megatron_init: false
```

## Verification

- 178 / 182 tests pass. The 4 failures are **pre-existing** and unrelated to this change:
  - `tests/jepa/test_corruption.py` — 2 failures: `test_n_segments`, `test_partial_last_segment_zero_padded` (segment-size calculation bug present before this work)
  - `tests/test_metrics.py` — 2 failures: `test_per_class_keys`, `test_ap_nan_when_single_class` (per-class AP keys missing from metrics implementation)
- All 22 model unit tests pass (`tests/jepa/test_model.py`).
- Full Lightning smoketest suite passes (`tests/jepa/test_lightning.py`).
- Classifier and finetune-module tests pass.

## Implementation checklist

- [x] RoPE in self-attention and cross-attention (static tokens exempt)
- [x] SwiGLU / GeGLU FFN, selectable via `activation` config key
- [x] Linear biases removed (attention projections, FFN projections, output heads)
- [x] Configurable local/global attention (`local_window_size`, `global_attention_every_n`)
- [x] Fast attention via `torch.nn.functional.scaled_dot_product_attention` (SDPA)
- [x] Megatron-Core-style weight initialisation
- [x] Additional static tokens predicted by predictor with learned query embeddings
- [x] Gradual model extension: `load_partial_checkpoint()` + `extend_from_checkpoint` config key
- [x] Teacher mode: `"ema"` (default) or `"frozen"` (no EMA updates)
- [x] Config files updated (`jepa_base.yaml`, `jepa_finetune_base.yaml`)
- [x] Reference documentation updated (`docs/jepa_pretraining.md`)
- [x] Backward compatibility with existing checkpoints and tests

## Possible further TODOs

- [ ] **Fix pre-existing test failures** — `test_n_segments` / `test_partial_last_segment_zero_padded` (segment size mismatch) and `test_per_class_keys` (per-class AP keys in `MultiLabelMetrics`)
- [ ] **Flash Attention 2** — replace SDPA with `flash_attn` for memory-efficient training on long sequences; SDPA already delegates to FlashAttention when available, but explicit `flash_attn` gives more control (e.g. `window_size` for local attention natively)
- [ ] **Sequence packing** — batch multiple short documents into one sequence for higher GPU utilisation; requires tracking document boundaries for the local/global attention mask
- [ ] **Stochastic depth (LayerDrop)** — randomly drop transformer layers during training for regularisation; useful when extending to >12 layers
- [x] **`torch.compile` verification** — verified: smoketest runs cleanly with `model.compile=true`; PyTorch wraps the model as `OptimizedModule`, training completes 10 steps without errors
- [x] **Example named configs for standard variants** — created `jepa_light_2_local.yaml` (local window=64, global every 3rd layer), `jepa_light_2_geglu.yaml` (GeGLU activation), `jepa_light_2_staged_8l.yaml` (8-layer staged extension from 6-layer checkpoint)
