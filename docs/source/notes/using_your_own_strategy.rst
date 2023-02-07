.. _using own strategy:

Creating your own active learning strategies with PyRelationAL
==============================================================

While PyRelationAL already implements multiple standard active learning strategies, it is not exhaustive.
However, users can easily define their own strategies by subclassing
:py:class:`pyrelational.strategies.abstract_strategy.Strategy`
and overriding :py:meth:`pyrelational.strategies.abstract_strategy.Strategy.__call__`
Let's look at some examples.


Implementing a mixed strategy
_____________________________

In the first example, we show how to implement a mixed strategy for a regression task using the least confidence scorer
for informativeness in combination with representative sampling. The active_learning_step is decomposed in two steps:
1) identifying a subset of the queryable set based on the least confidence score (make sure that this subset is
sufficiently larger than the number of annotations we want to query) and 2) select representative samples from this
subset based on euclidean distance between input features.

.. code-block:: python

    import torch
    from pyrelational.informativeness import regression_least_confidence
    from pyrelational.informativeness.task_agnostic import representative_sampling
    from pyrelational.strategies.generic_al_strategy import Strategy


    class MixedStrategy(Strategy):
        """
        Implements a strategy that combines least_confidence scorer with representative sampling.
        To this end, 10 times more samples than requested are selected based on least_confidence scorer,
        the list is then reduced based on representative_sampling.
        """

        def __init(self, datamanager, model):
            super(MixedStrategy, self).__init__(datamanager, model)

        def __call__(self, num_annotate):
            self.model.train(self.l_loader, self.valid_loader)
            output = self.model(self.u_loader)
            scores = regression_least_confidence(x=output)
            ixs = torch.argsort(scores, descending=True).tolist()
            ixs = [self.u_indices[i] for i in ixs[: 10 * num_annotate]]
            subquery = torch.stack(self.data_manager.get_sample_feature_vectors(ixs))
            new_ixs = representative_sampling(subquery)
            return [ixs[i] for i in new_ixs]

Implementing an :math:`\epsilon`-greedy strategy
________________________________________________

In the second example, we implement an :math:`\epsilon`-greedy strategy: for :math:`N` queries, :math:`(1-\epsilon)N`
are selected greedily based on model prediction and :math:`\epsilon N` are selected uniformly at
random from the remaining queryable set.

.. code-block:: python

    import torch
    import numpy as np
    from pyrelational.informativeness import regression_mean_prediction
    from pyrelational.strategies.generic_al_strategy import Strategy


    class EpsilonGreedyStrategy(Strategy):
        """
        Implements an epsilon-greedy strategy, whereby a percentage of the samples to annotate
        are selected randomly while the remaining are selected greedily.
        """

        def __init(self, datamanager, model):
            super(EpsilonGreedyStrategy, self).__init__(datamanager, model)

        def __call__(self, num_annotate, eps=0.05):
            assert 0 <= eps <= 1, "epsilon should be a float between 0 and 1"
            self.model.train(self.l_loader, self.valid_loader)
            output = self.model(self.u_loader)
            scores = regression_mean_prediction(x=output)
            ixs = torch.argsort(scores, descending=True).tolist()
            greedy_annotate = int((1-eps)*num_annotate)
            ixs = [self.u_indices[i] for i in ixs[: greedy_annotate]]
            remaining_u_indices = list(set(self.u_indices) - set(ixs))
            random_annotate = np.random.choice(remaining_u_indices, num_annotate-greedy_annotate, replace=False)
            return ixs + random_annotate.tolist()

See the `examples folder <https://github.com/RelationRx/pyrelational/examples>`_ in the source repository for more examples.
