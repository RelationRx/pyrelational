"""Active learning using ratio based confidence uncertainty measure."""

from pyrelational.informativeness import RatioConfidence
from pyrelational.samplers import DeterministicSampler
from pyrelational.strategies.classification.classification_strategy import (
    ClassificationStrategy,
)


class RatioConfidenceStrategy(ClassificationStrategy):
    """Implements Ratio Confidence Strategy.

    Unlabelled samples are scored and queried based on the ratio confidence for classification scorer.
    """

    def __init__(self, axis: int = -1):
        """Initialize the strategy with the ratio confidence scorer and a deterministic scorer for classification."""
        super().__init__(RatioConfidence(axis=axis), DeterministicSampler())
