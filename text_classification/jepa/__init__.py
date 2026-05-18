"""JEPA byte-segment pretraining and finetuning for tokenizer-free text encoders."""
from text_classification.jepa.corruption import (
    CorruptionConfig,
    CorruptionType,
    PAD_BYTE,
    BYTE_VOCAB_SIZE,
    SEGMENT_SIZE,
    generate_student_view,
    text_to_canonical_segments,
)
from text_classification.jepa.model import ByteSegmentJEPA, ByteSegmentEncoder
from text_classification.jepa.loss import LossWeights, compute_total_loss
from text_classification.jepa.lightning_module import JEPAPretrainingModule
from text_classification.jepa.classifier import JEPAClassifier
from text_classification.jepa.finetune_data import (
    JEPAClassificationDataset,
    JEPAClassificationDataModule,
    jepa_collate_fn,
)
from text_classification.jepa.finetune_module import JEPAFinetuneModule

__all__ = [
    "ByteSegmentJEPA",
    "ByteSegmentEncoder",
    "CorruptionConfig",
    "CorruptionType",
    "JEPAClassificationDataModule",
    "JEPAClassificationDataset",
    "JEPAClassifier",
    "JEPAFinetuneModule",
    "JEPAPretrainingModule",
    "LossWeights",
    "PAD_BYTE",
    "BYTE_VOCAB_SIZE",
    "SEGMENT_SIZE",
    "compute_total_loss",
    "generate_student_view",
    "jepa_collate_fn",
    "text_to_canonical_segments",
]
