"""Unit tests for active learning manager
"""
from typing import Type
from unittest import TestCase

import pytest
from parameterized import parameterized_class

from pyrelational.model_managers.mcdropout_model_manager import (
    LightningMCDropoutModelManager,
)
from pyrelational.pipeline import Pipeline
from pyrelational.strategies import Strategy
from pyrelational.strategies.classification import (
    EntropyClassificationStrategy,
    LeastConfidenceStrategy,
    MarginalConfidenceStrategy,
    RatioConfidenceStrategy,
)
from pyrelational.strategies.regression import (
    BALDStrategy,
    ExpectedImprovementStrategy,
    MeanPredictionStrategy,
    SoftBALDStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
)
from tests.test_utils import (
    BreastCancerClassifier,
    DiabetesRegressionModel,
    get_classification_dataset,
    get_regression_dataset,
)

REGRESSION_STRATEGIES = [
    BALDStrategy,
    ExpectedImprovementStrategy,
    MeanPredictionStrategy,
    SoftBALDStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
]
CLASSIFICATION_STRATEGIES = [
    EntropyClassificationStrategy,
    LeastConfidenceStrategy,
    MarginalConfidenceStrategy,
    RatioConfidenceStrategy,
]


@parameterized_class(
    [{"run_type": "regression", "strategy_class": strategy} for strategy in REGRESSION_STRATEGIES]
    + [{"run_type": "classification", "strategy_class": strategy} for strategy in CLASSIFICATION_STRATEGIES]
)
class TestPipeline(TestCase):
    """Class containing tests for active learning strategies."""

    strategy_class: Type[Strategy]

    def setUp(self) -> None:
        """Set up attributes."""
        if self.run_type == "regression":
            model_manager = LightningMCDropoutModelManager(DiabetesRegressionModel, {"ensemble_size": 3}, {"epochs": 1})
            self.datamanager = get_regression_dataset(hit_ratio_at=5)
        else:
            model_manager = LightningMCDropoutModelManager(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
            self.datamanager = get_classification_dataset(hit_ratio_at=5)
        strategy = self.strategy_class()
        self.pipeline = Pipeline(data_manager=self.datamanager, model_manager=model_manager, strategy=strategy)

    def test_performances(self) -> None:
        """Test theoretical and current performance returns."""
        self.pipeline.compute_theoretical_performance()
        self.assertIn("full", self.pipeline.performances)
        self.assertEqual(len(list(self.pipeline.performances.keys())), 1)

    def test_full_active_learning_run(self) -> None:
        """Test that full run completes and attributes have the expected shapes."""
        self.pipeline.compute_theoretical_performance()
        self.pipeline.run(num_annotate=200)
        # Test performance history data frame
        df = self.pipeline.summary()
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(len(self.pipeline.data_manager.l_indices), len(self.datamanager.train_indices))
        self.assertEqual(len(self.pipeline.data_manager.u_indices), 0)
        self.assertEqual({"full", 0, 1, 2}, set(list(self.pipeline.performances.keys())))
        for k in {"full", 0, 1, 2}:
            self.assertIn("hit_ratio", self.pipeline.performances[k].keys())

    def test_get_percentage_labelled(self) -> None:
        """Test get percentage labelled method return correct output."""
        self.assertEqual(self.pipeline.percentage_labelled, pytest.approx(10, 5))

    def test_get_dataset_size(self) -> None:
        """Test get dataset size."""
        self.assertEqual(self.pipeline.dataset_size, len(self.datamanager))
