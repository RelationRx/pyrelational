"""Active learning using Entropy uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle.
"""

import numpy as np
import torch
from torch import Tensor

from pyrelational.informativeness import object_detection_entropy
from pyrelational.strategies.object_detection.abstract_object_detection_strategy import (
    ObjectDetectionStrategy,
)


class EntropyObjectDetectionStrategy(ObjectDetectionStrategy):
    """Implements Entropy Strategy whereby unlabelled samples are scored and queried based on
    the entropy for object detection scorer"""

    def __init__(self, aggregation_type: str = "max") -> None:
        super().__init__(aggregation_type)

    def scoring_function(
        self, predictions: Tensor, aggregation_type: str = "max"
    ) -> Tensor:
        return object_detection_entropy(
            predictions, aggregation_type=aggregation_type
        )
