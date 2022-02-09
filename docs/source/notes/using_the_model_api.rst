.. _build your own model:

Defining learning models compatible with pyrelational
===================================================

To interact with the pyrelational library, models need to be wrapped within a pyrelational module (:mod:`pyrelational.models`)
that defines required methods for instantiation, training, and testing.



Using a pytorch module
______________________

Let's first look an an example wrapper for a simple pytorch Module

.. code-block:: python

   import torch
   import torch.nn as nn
   from pyrelational.models.mcdropout_model import GenericMCDropoutModel
   class TorchModuleWrapper(GenericMCDropoutModel):
       def __init__(self,
           model_class,
           model_config,
           trainer_config,
           n_estimators= 5,
           eval_dropout_prob= 0.2,
       ):
           super(Wrapper, self).__init__(model_class,model_config,trainer_config,n_estimators,eval_dropout_prob)

       def train(self, train_loader, val_loader=None):
           model = self.init_model()
           criterion = nn.MSELoss()
           optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_config["lr"])
           for _ in range(self.trainer_config["epochs"]):
               for x, y in train_loader:
                   optimizer.zero_grad()
                   out = model(x)
                   loss = criterion(out, y)
                   loss.backward()
                   optimizer.step()
           self.current_model = model # store current model

       def test(self, loader):
           if self.current_model is None:
               raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
           criterion = nn.MSELoss()
           with torch.no_grad():
               tst_loss, cnt = 0, 0
               for x, y in loader:
                   loss = criterion(self.current_model(x), y)
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

pyrelational implements default classes (see :py:meth:`pyrelational.models.lightning_model.LightningModel`) relying on
pytorch lightning as the Trainer class offload much of the training routine definition to pytorch lighntning.
For example, users can create ensembles of pytorch lightning modules directly as

.. code-block:: python

    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score
    from pytorch_lightning.core.lightning import LightningModule
    from pyrelational.models.ensemble_model import LightningEnsembleModel

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

    # step 2: wrap the model in a LightningEnsembleModel
    wrapper = LightningEnsembleModel(
                  DigitClassifier,
                  {"dropout_rate":0.1, "lr":3e-4},
                  {"epochs":1,"gpus":1},
                  n_estimators=5,
            )

See the `examples folder <https://github.com/RelationRx/pyrelational/examples>`_ in the source repository for more examples.
