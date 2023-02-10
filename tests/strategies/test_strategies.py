from typing import Any, Dict, Type
from unittest import TestCase

from parameterized import parameterized_class

from pyrelational.model_managers.mcdropout_model_manager import (
    LightningMCDropoutModelManager,
)
from pyrelational.strategies import Strategy
from tests.strategies.agnostic_strategy_test_cases import TASK_AGNOSTIC_TEST_CASES
from tests.strategies.classification_strategy_test_cases import (
    CLASSIFICATION_TEST_CASES,
)
from tests.strategies.regression_strategy_test_cases import REGRESSION_TEST_CASES
from tests.test_utils import (
    BreastCancerClassifier,
    DiabetesRegressionModel,
    get_classification_dataset,
    get_regression_dataset,
)


@parameterized_class(TASK_AGNOSTIC_TEST_CASES + CLASSIFICATION_TEST_CASES + REGRESSION_TEST_CASES)
class TestStrategies(TestCase):
    """Class containing unit test of strategies."""

    task_type: str
    strategy_class: Type[Strategy]
    strategy_kwargs: Dict[str, Any]

    def setUp(self) -> None:
        """Define model and datamanager."""
        if self.task_type == "regression":
            model_class = DiabetesRegressionModel
            self.datamanager = get_regression_dataset()
        else:
            model_class = BreastCancerClassifier
            self.datamanager = get_classification_dataset()
        self.model_manager = LightningMCDropoutModelManager(
            model_class,
            {"ensemble_size": 3},
            {"epochs": 5, "gpus": 0},
        )
        self.strategy = self.strategy_class(**self.strategy_kwargs)

    def test_suggest(self) -> None:
        """Test suggest return the required number of sample indices."""
        out = self.strategy.suggest(num_annotate=5, model_manager=self.model_manager, data_manager=self.datamanager)
        self.assertEqual(len(out), 5)

    def test_str_print(self) -> None:
        """Check str returns expected string"""
        self.assertEqual(str(self.strategy), f"Strategy: {self.strategy.__class__.__name__}")
