"""Active learning using entropy based confidence uncertainty measure.

The score is computed between classes in the posterior predictive distribution to
choose which observations to propose to the oracle.
"""

from pyrelational.batch_mode_samplers import TopKSampler
from pyrelational.informativeness import Entropy
from pyrelational.strategies.classification.classification_strategy import (
    ClassificationStrategy,
)


class EntropyClassificationStrategy(ClassificationStrategy):
    """Implements Entropy Classification Strategy."""

    def __init__(self, axis: int = -1):
        """Initialise the strategy with entropy scorer and deterministic sampler."""
        super().__init__(Entropy(axis=axis), TopKSampler())
