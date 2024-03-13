"""
Active learning using least confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""

import numpy as np
import torch
from torch import Tensor

from ids_research.strategies.informativeness import (
    object_detection_least_confidence,
)
from ids_research.strategies.object_detection.abstract_object_detection_strategy import (
    ObjectDetectionStrategy,
)


class LeastConfidenceStrategy(ObjectDetectionStrategy):
    """Implements Least Confidence Strategy whereby unlabelled samples are scored and queried based on
    the least confidence for object detection scorer"""

    def __init__(self, aggregation_type: str = "max") -> None:
        super().__init__(aggregation_type)

    def scoring_function(
        self, predictions: Tensor, aggregation_type: str = "max"
    ) -> Tensor:
        return object_detection_least_confidence(
            predictions, aggregation_type=aggregation_type
        )
