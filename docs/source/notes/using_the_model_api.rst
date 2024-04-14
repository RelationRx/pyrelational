.. _build your own model:

Defining learning models compatible with PyRelationAL
=====================================================

To interact with the PyRelationAL library, models need to be wrapped within a PyRelationAL model manager (:mod:`pyrelational.model_managers`)
that defines required methods for instantiation, training, and testing (optionally). Below are a couple of examples of specifying model model_manager
with model implementations based in common ML frameworks such as PyTorch, Scikit-Learn, Pytorch Lightning

Using a pytorch module
______________________

Let's look at an example model manager for a simple pytorch Module

.. code-block:: python

   import torch
   import torch.nn as nn
   from pyrelational.model_managers.mcdropout_model import MCDropoutModelManager


   class TorchModuleWrapper(MCDropoutModelManager):
       def __init__(self,
           model_class,
           model_config,
           trainer_config,
           n_estimators= 5,
           eval_dropout_prob= 0.2,
       ):
           super(Wrapper, self).__init__(model_class,model_config,trainer_config,n_estimators,eval_dropout_prob)

       def train(self, train_loader, val_loader=None):
           model = self._init_model()
           criterion = nn.MSELoss()
           optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_config["lr"])
           for _ in range(self.trainer_config["epochs"]):
               for x, y in train_loader:
                   optimizer.zero_grad()
                   out = model(x)
                   loss = criterion(out, y)
                   loss.backward()
                   optimizer.step()
           self._current_model = model # store current model

       def test(self, loader):
           if not self.is_trained():
               raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
           criterion = nn.MSELoss()
           self._current_model.eval()
           with torch.no_grad():
               tst_loss, cnt = 0, 0
               for x, y in loader:
                   loss = criterion(self._current_model(x), y)
                   cnt += x.size(0)
                   tst_loss += loss.item()
           return {"test_loss": tst_loss/cnt}

This can now be instantiated with any pytorch module, for instance

.. code-block:: python

   model = TorchModuleWrapper(
                     nn.Linear,
                     {"in_features": 5, "out_features":1},
                     {"epochs":100, "lr":3e-4},
                     n_estimators=10,
                  )

Using pytorch lightning module
______________________________

PyRelationAL implements default classes (see :py:meth:`pyrelational.model_managers.lightning_model.LightningModelManager`) relying on
pytorch lightning as the Trainer class offload much of the training routine definition to pytorch lighntning.
For example, users can create ensembles of pytorch lightning modules directly as

.. code-block:: python

    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score
    from lightning.pytorch import LightningModule
    from pyrelational.model_managers.ensemble_model_manager import LightningEnsembleModelManager

    # step 1: define the LightningModule with necessary methods
    class DigitClassifier(LightningModule):
        """Custom module for a simple convnet classifier"""

        def __init__(self, dropout_rate=0, lr=3e-4):
            super(DigitClassifier, self).__init__()
            self.layer_1 = nn.Linear(8*8, 16)
            self.layer_2 = nn.Linear(16, 32)
            self.dropout = nn.Dropout(dropout_rate)
            self.layer_3 = nn.Linear(32, 10)
            self.lr = lr

        def forward(self, x):
            x = self.layer_1(x)
            x = F.relu(x)
            x = self.layer_2(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.layer_3(x)
            x = F.log_softmax(x, dim=1)
            return x

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            self.log("loss", loss.item())
            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            self.log("test_loss", loss)

            # compute accuracy
            _, y_pred = torch.max(logits.data, 1)
            accuracy = accuracy_score(y, y_pred)
            self.log("accuracy", accuracy)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    # step 2: wrap the model in a LightningEnsembleModelManager
    wrapper = LightningEnsembleModelManager(
                  DigitClassifier,
                  {"dropout_rate":0.1, "lr":3e-4},
                  {"epochs":1,"accelerator":"gpu", "devices":1},
                  n_estimators=5,
            )


Using a scikit-learn module
___________________________

Let's look at an example model manager based on an ensemble of k-fold SVMs regressors, each implemented in
scikit-learn. This could be used in query-by-committee based strategies.

.. code-block:: python

    class ScikitSVMEnsemble(ModelManager):
        """Custom ModelManager for an ensemble of K-fold SVM regressors, as found in query-by-commitee
        strategies. Can easily be adapted to any other regressor

        Args:
        model_class (sklearn estimator): Estimator that should be ensembled (e.g. MLPRegressor)
        num_estimators (int): number of estimators in the ensemble
        model_config (dict): dictionary containing any model_class specific arguments
        trainer_config (dict): dictionary containing any taining specific arguments
        """

    def __init__(self, model_class, num_estimators, model_config, trainer_config):
        super(EnsembleScikit, self).__init__(model_class, model_config, trainer_config)
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.num_estimators = num_estimators

    def train(self, train_loader):
        """
        Args:
        train_loader (torch.utils.data.DataLoader): A torch dataloader with a numpy compatible collate function
            you can also just adapt this to something else.
        """
        train_x, train_y = next(iter(train_loader)) # assumes dataloader returns full set of available observations
        estimators = []

        k = self.num_estimators
        kf = KFold(n_splits=k)
        from joblib import Parallel, delayed
        estimators = Parallel(n_jobs=-1)(delayed(self._init_model().fit)(train_x[train_index], train_y[train_index]) for train_index, _ in kf.split(train_x))

        # Set the current model to the list of trained regressors
        self._current_model = estimators

    def test(self, loader):
        if not self.is_trained():
            raise ValueError(
                "No current model, call 'train(X, y)' to train the model first"
            )
        X, y = next(iter(loader))
        scores = []
        for idx in range(self.num_estimators):
            estimator = self._current_model[idx]
            predictions = estimator.predict(X)
            score = mean_squared_error(y, predictions)
            scores.append(score)
        return {"MSE": np.mean(scores)}

    def __call__(self, loader):
        if not self.is_trained():
            raise ValueError(
                "No current model, call 'train(X, y)' to train the model first"
            )
        X, _ = next(iter(loader))
        predictions = []  # list of num_estimator predictions of shape y
        for est_idx in range(self.num_estimators):
            estimator = self._current_model[est_idx]
            predictions.append(torch.FloatTensor(estimator.predict(X)))
        predictions = torch.vstack(predictions)
        return predictions

See the `examples folder <https://github.com/RelationRx/pyrelational/examples>`_ in the source repository for more examples.
