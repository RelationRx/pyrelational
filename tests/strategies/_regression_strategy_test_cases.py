from pyrelational.strategies.regression import (
    AverageScoreStrategy,
    BALDStrategy,
    ExpectedImprovementStrategy,
    SoftBALDStrategy,
    StandardDeviationStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
)

REGRESSION_TEST_CASES = [
    {"task_type": "regression", "strategy_class": BALDStrategy, "strategy_kwargs": {}},
    {"task_type": "regression", "strategy_class": StandardDeviationStrategy, "strategy_kwargs": {}},
    {
        "task_type": "regression",
        "strategy_class": ExpectedImprovementStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "regression",
        "strategy_class": AverageScoreStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "regression",
        "strategy_class": SoftBALDStrategy,
        "strategy_kwargs": {},
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
