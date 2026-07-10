"""Reusable experiment pipeline helpers for Segment Every Grain.

The helpers in this package keep the research workflow split into small,
testable stages:

1. fit or store synthetic-noise parameters,
2. generate synthetic noisy image/mask pairs,
3. build an explicit training set from clean/synthetic/real sources,
4. train or fine-tune a U-Net,
5. evaluate a trained model on a held-out image/mask directory.
"""

__all__ = []
