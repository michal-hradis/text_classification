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
