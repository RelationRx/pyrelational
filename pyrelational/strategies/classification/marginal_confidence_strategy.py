"""Active learning using marginal confidence uncertainty measure."""

from pyrelational.batch_mode_samplers import TopKSampler
from pyrelational.informativeness import MarginConfidence
from pyrelational.strategies.classification.classification_strategy import (
    ClassificationStrategy,
)


class MarginalConfidenceStrategy(ClassificationStrategy):
    """Implements Marginal Confidence Strategy.

    Unlabelled samples are scored and queried based on the marginal confidence for classification scorer.
    """

    def __init__(self, axis: int = -1):
        """Initialize the strategy with the marginal confidence scorer and a deterministic scorer for classification."""
        super().__init__(MarginConfidence(axis=axis), TopKSampler())
