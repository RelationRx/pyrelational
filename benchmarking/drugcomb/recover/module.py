from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection, PearsonCorrCoef, R2Score

from benchmarking.drugcomb.recover.model import BilinearFiLMMLPPredictor


class RecoverModel(LightningModule):

    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection

    def __init__(
        self,
        drugs_dim: int,
        cell_lines_dim: int,
        encoder_layer_dims: List[int],
        decoder_layer_dims: List[int],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_samples: int = 100,
        exponent_cap: float = 80,
    ) -> None:
        super().__init__()
        self.mu_predictor = BilinearFiLMMLPPredictor(drugs_dim, cell_lines_dim, encoder_layer_dims, decoder_layer_dims)
        self.std_predictor = BilinearFiLMMLPPredictor(drugs_dim, cell_lines_dim, encoder_layer_dims, decoder_layer_dims)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.exponent_cap = exponent_cap
        self.set_metrics()

    def set_metrics(self) -> None:
        for split in ["train", "val", "test"]:
            setattr(
                self,
                f"{split}_metrics",
                MetricCollection({"mse": MeanSquaredError(), "r2": R2Score(), "pearson": PearsonCorrCoef()}),
            )

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        drugs_a, drugs_b, cell_lines, synergies = batch
        predicted_mean, predicted_log_sigma2 = self.forward(drugs_a, drugs_b, cell_lines)
        self.train_metrics.update(preds=predicted_mean.flatten(), target=synergies.flatten())
        loss = self.loss(predicted_mean, predicted_log_sigma2, synergies)
        self.log("loss", loss.item(), prog_bar=True)
        return loss

    def loss(self, predicted_mean: Tensor, predicted_log_sigma2: Tensor, synergies: Tensor) -> Tensor:
        """The mu_predictor model is trained using MSE while the std_predictor is trained using
        the adaptive NLL criterion."""
        mse = (predicted_mean - synergies) ** 2
        denom = 1e-8 + 2 * torch.exp(
            torch.min(
                predicted_log_sigma2, torch.tensor(self.exponent_cap, dtype=synergies.dtype, device=synergies.device)
            )  # exponent capped for stability
        )
        return torch.mean(mse) + torch.mean(predicted_log_sigma2 / 2 + mse.detach() / denom)

    def forward(self, drugs_a: Tensor, drugs_b: Tensor, cell_lines: Tensor) -> Tuple[Tensor, Tensor]:
        return self.mu_predictor(drugs_a, drugs_b, cell_lines), self.std_predictor(drugs_a, drugs_b, cell_lines)

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        drugs_a, drugs_b, cell_lines, synergies = batch
        predicted_mean, _ = self.forward(drugs_a, drugs_b, cell_lines)
        self.val_metrics.update(preds=predicted_mean.flatten(), target=synergies.flatten())

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        drugs_a, drugs_b, cell_lines, synergies = batch
        predicted_mean, _ = self.forward(drugs_a, drugs_b, cell_lines)
        self.test_metrics.update(preds=predicted_mean.flatten(), target=synergies.flatten())

    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        drugs_a, drugs_b, cell_lines, _ = batch
        predicted_mean, predicted_log_sigma2 = self.forward(drugs_a, drugs_b, cell_lines)
        return predicted_mean.unsqueeze(0) + torch.randn(
            (self.num_samples,) + tuple(predicted_mean.shape), device=drugs_a.device
        ) * torch.exp(predicted_log_sigma2 / 2)

    def on_train_epoch_end(self) -> None:
        self.compute_and_log_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self.compute_and_log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self.compute_and_log_metrics("test")

    def compute_and_log_metrics(self, split: str) -> STEP_OUTPUT:
        metrics = getattr(self, f"{split}_metrics")
        scores: Dict[str, Any] = metrics.compute()
        self.log_dict(scores)
        metrics.reset()
        return scores

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
