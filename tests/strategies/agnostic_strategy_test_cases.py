from sklearn.cluster import AgglomerativeClustering

from pyrelational.strategies.task_agnostic import (
    RandomAcquisitionStrategy,
    RelativeDistanceStrategy,
    RepresentativeSamplingStrategy,
)

TASK_AGNOSTIC_TEST_CASES = [
    {"task_type": "regression", "strategy_class": RandomAcquisitionStrategy, "strategy_kwargs": {}},
    {"task_type": "regression", "strategy_class": RelativeDistanceStrategy, "strategy_kwargs": {}},
    {
        "task_type": "regression",
        "strategy_class": RelativeDistanceStrategy,
        "strategy_kwargs": {"metric": "cosine"},
    },
    {
        "task_type": "regression",
        "strategy_class": RepresentativeSamplingStrategy,
        "strategy_kwargs": {"clustering_method": "AffinityPropagation"},
    },
    {
        "task_type": "regression",
        "strategy_class": RepresentativeSamplingStrategy,
        "strategy_kwargs": {"clustering_method": AgglomerativeClustering(n_clusters=10)},
    },
]
