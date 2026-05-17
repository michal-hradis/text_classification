#!/usr/bin/env python3
"""Convenience wrapper for JEPA pretraining (repository root).

Delegates to :mod:`text_classification.jepa.train`.

Usage::

    python jepa_pretrain.py configs/jepa_base.yaml
    python jepa_pretrain.py configs/jepa_smoketest.yaml model.seg_dim=128
"""
from text_classification.jepa.train import main

if __name__ == "__main__":
    main()
