from pyrelational.strategies.regression import (
    BALDStrategy,
    ExpectedImprovementStrategy,
    LeastConfidenceStrategy,
    MeanPredictionStrategy,
    SoftBALDStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
)

REGRESSION_TEST_CASES = [
    {"task_type": "regression", "strategy_class": BALDStrategy, "strategy_kwargs": {}},
    {"task_type": "regression", "strategy_class": LeastConfidenceStrategy, "strategy_kwargs": {}},
    {
        "task_type": "regression",
        "strategy_class": ExpectedImprovementStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "regression",
        "strategy_class": MeanPredictionStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "regression",
        "strategy_class": SoftBALDStrategy,
        "strategy_kwargs": {"temperature": 0.42},
    },
    {
        "task_type": "regression",
        "strategy_class": ThompsonSamplingStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "regression",
        "strategy_class": UpperConfidenceBoundStrategy,
        "strategy_kwargs": {"kappa": 0.42},
    },
]
