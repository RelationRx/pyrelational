from pyrelational.strategies.regression import (
    BALDStrategy,
    ExpectedImprovementStrategy,
    GreedyStrategy,
    SoftBALDStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
    VarianceReductionStrategy,
)

REGRESSION_TEST_CASES = [
    {"task_type": "regression", "strategy_class": BALDStrategy, "strategy_kwargs": {}},
    {"task_type": "regression", "strategy_class": VarianceReductionStrategy, "strategy_kwargs": {}},
    {
        "task_type": "regression",
        "strategy_class": ExpectedImprovementStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "regression",
        "strategy_class": GreedyStrategy,
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
