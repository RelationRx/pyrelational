"""
Active learning using marginal confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""

from torch import Tensor

from pyrelational.informativeness import (
    object_detection_margin_confidence,
)
from pyrelational.strategies.object_detection.abstract_object_detection_strategy import (
    ObjectDetectionStrategy,
)


class MarginalConfidenceStrategy(ObjectDetectionStrategy):
    """Implements Marginal Confidence Strategy whereby unlabelled samples are scored and queried based on
    the marginal confidence for object detection scorer"""

    def __init__(self, aggregation_type: str = "max") -> None:
        super().__init__(aggregation_type)

    def scoring_function(
        self, predictions: Tensor, aggregation_type: str = "max"
    ) -> Tensor:
        return object_detection_margin_confidence(
            predictions, aggregation_type=aggregation_type
        )
