"""Active learning using least confidence uncertainty measure."""

from pyrelational.informativeness import LeastConfidence
from pyrelational.samplers import DeterministicSampler
from pyrelational.strategies.classification.classification_strategy import (
    ClassificationStrategy,
)


class LeastConfidenceStrategy(ClassificationStrategy):
    """Implements Least Confidence Strategy.

    Unlabelled samples are scored and queried based on the least confidence for classification scorer.
    """

    def __init__(self, axis: int = -1):
        """Initialize the strategy with the least confidence scorer and a deterministic scorer for classification."""
        super().__init__(LeastConfidence(axis=axis), DeterministicSampler())
