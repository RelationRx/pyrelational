from pyrelational.strategies.classification import (
    EntropyClassificationStrategy,
    LeastConfidenceStrategy,
    MarginalConfidenceStrategy,
    RatioConfidenceStrategy,
)

CLASSIFICATION_TEST_CASES = [
    {
        "task_type": "classification",
        "strategy_class": EntropyClassificationStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "classification",
        "strategy_class": LeastConfidenceStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "classification",
        "strategy_class": MarginalConfidenceStrategy,
        "strategy_kwargs": {},
    },
    {
        "task_type": "classification",
        "strategy_class": RatioConfidenceStrategy,
        "strategy_kwargs": {},
    },
]
