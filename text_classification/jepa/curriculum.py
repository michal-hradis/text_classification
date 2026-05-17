"""Curriculum schedule for JEPA pretraining (spec §9).

The curriculum controls four training dimensions:
- Document length (characters)
- Missing-segment severity
- Byte/character noise severity
- Foreign-replacement rate

Training advances through stages by monitoring validation loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from text_classification.jepa.corruption import CorruptionConfig


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """Single curriculum stage configuration."""
    name: str
    min_chars: int
    max_chars: int
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)


# ---------------------------------------------------------------------------
# Predefined stages (spec §9.1)
# ---------------------------------------------------------------------------

CURRICULUM_STAGES: list[CurriculumStage] = [
    CurriculumStage(
        name="sanity",
        min_chars=500,
        max_chars=1_000,
        corruption=CorruptionConfig(
            missing_span_prob=0.10,
            missing_span_frac_min=0.05,
            missing_span_frac_max=0.15,
            segment_dropout_prob=0.02,
            foreign_replace_frac=0.00,
            byte_corrupt_prob=0.01,
            truncate_prob=0.05,
            truncate_frac_max=0.10,
        ),
    ),
    CurriculumStage(
        name="short",
        min_chars=500,
        max_chars=2_000,
        corruption=CorruptionConfig(
            missing_span_prob=0.20,
            missing_span_frac_min=0.10,
            missing_span_frac_max=0.25,
            segment_dropout_prob=0.05,
            foreign_replace_frac=0.02,
            byte_corrupt_prob=0.04,
            truncate_prob=0.10,
            truncate_frac_max=0.15,
        ),
    ),
    CurriculumStage(
        name="medium",
        min_chars=1_000,
        max_chars=5_000,
        corruption=CorruptionConfig(
            missing_span_prob=0.30,
            missing_span_frac_min=0.20,
            missing_span_frac_max=0.40,
            segment_dropout_prob=0.05,
            foreign_replace_frac=0.05,
            byte_corrupt_prob=0.07,
            truncate_prob=0.10,
            truncate_frac_max=0.20,
        ),
    ),
    CurriculumStage(
        name="long",
        min_chars=3_000,
        max_chars=10_000,
        corruption=CorruptionConfig(
            missing_span_prob=0.40,
            missing_span_frac_min=0.30,
            missing_span_frac_max=0.55,
            segment_dropout_prob=0.08,
            foreign_replace_frac=0.10,
            byte_corrupt_prob=0.12,
            truncate_prob=0.15,
            truncate_frac_max=0.20,
        ),
    ),
    CurriculumStage(
        name="full",
        min_chars=500,
        max_chars=10_000,
        corruption=CorruptionConfig(
            missing_span_prob=0.50,
            missing_span_frac_min=0.35,
            missing_span_frac_max=0.60,
            segment_dropout_prob=0.10,
            foreign_replace_frac=0.15,
            byte_corrupt_prob=0.15,
            truncate_prob=0.15,
            truncate_frac_max=0.20,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Advances through :data:`CURRICULUM_STAGES` when validation loss plateaus.

    Args:
        stages:   Ordered list of :class:`CurriculumStage` objects.
        patience: Number of validation runs without improvement before
                  advancing to the next stage.
    """

    def __init__(
        self,
        stages: list[CurriculumStage] | None = None,
        patience: int = 3,
    ) -> None:
        self.stages = stages if stages is not None else CURRICULUM_STAGES
        self.patience = patience
        self._idx: int = 0
        self._best: float = float("inf")
        self._no_improve: int = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    @property
    def stage_idx(self) -> int:
        return self._idx

    def report_val_loss(self, val_loss: float) -> bool:
        """Report a validation loss value.

        Returns:
            ``True`` if the scheduler advanced to the next stage,
            ``False`` otherwise.
        """
        if val_loss < self._best - 1e-6:
            self._best = val_loss
            self._no_improve = 0
            return False
        self._no_improve += 1
        if self._no_improve >= self.patience and self._idx < len(self.stages) - 1:
            self._idx += 1
            self._no_improve = 0
            self._best = float("inf")
            return True
        return False
