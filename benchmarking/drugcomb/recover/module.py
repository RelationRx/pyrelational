from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection, PearsonCorrCoef, R2Score

from .model import BilinearFiLMMLPPredictor


class RECOVER(LightningModule):

    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection

    def __init__(
        self, drugs_dim: int, cell_lines_dim: int, encoder_layer_dims: List[int], decoder_layer_dims: List[int]
    ) -> None:
        super().__init__()
        self.mu_predictor = BilinearFiLMMLPPredictor(drugs_dim, cell_lines_dim, encoder_layer_dims, decoder_layer_dims)
        self.std_predictor = BilinearFiLMMLPPredictor(drugs_dim, cell_lines_dim, encoder_layer_dims, decoder_layer_dims)

    def set_metrics(self) -> None:
        for split in ["train", "val", "test"]:
            setattr(
                self,
                f"{split}_metrics",
                MetricCollection({"mse": MeanSquaredError(), "r2": R2Score(), "pearson": PearsonCorrCoef()}),
            )

    def training_step(self, drugs_a: Tensor, drugs_b: Tensor, cell_lines: Tensor, synergies: Tensor) -> STEP_OUTPUT:
        predicted_mean, predicted_log_sigma2 = self.forward(drugs_a, drugs_b, cell_lines)
        self.train_metrics.update(preds=predicted_mean, target=synergies)
        return self.loss(predicted_mean, predicted_log_sigma2, synergies)

    def loss(self, predicted_mean: Tensor, predicted_log_sigma2: Tensor, synergies: Tensor) -> Tensor:
        """The mu_predictor model is trained using MSE while the std_predictor is trained using
        the adaptive NLL criterion."""
        mse = (predicted_mean - synergies) ** 2
        denom = 2 * torch.exp(
            torch.min(
                predicted_log_sigma2, torch.tensor(80, dtype=torch.float32).to(synergies.device)
            )  # exponent capped at 80 for stability
        )
        return torch.mean(mse) + torch.mean(predicted_log_sigma2 / 2 + mse.detach() / denom)

    def forward(self, drugs_a: Tensor, drugs_b: Tensor, cell_lines: Tensor) -> Tuple[Tensor, Tensor]:
        return self.mu_predictor(drugs_a, drugs_b, cell_lines), self.std_predictor(drugs_a, drugs_b, cell_lines)

    def validation_step(self, drugs_a: Tensor, drugs_b: Tensor, cell_lines: Tensor, synergies: Tensor) -> None:
        predicted_mean, _ = self.forward(drugs_a, drugs_b, cell_lines)
        self.val_metrics.update(preds=predicted_mean, target=synergies)

    def test_step(self, drugs_a: Tensor, drugs_b: Tensor, cell_lines: Tensor, synergies: Tensor) -> None:
        predicted_mean, _ = self.forward(drugs_a, drugs_b, cell_lines)
        self.test_metrics.update(preds=predicted_mean, target=synergies)

    def predict_step(self, drugs_a: Tensor, drugs_b: Tensor, cell_lines: Tensor) -> Tensor:
        predicted_mean, predicted_log_sigma2 = self.forward(drugs_a, drugs_b, cell_lines)
        return predicted_mean + torch.randn((len(predicted_mean), 100)).to(drugs_a.device) * torch.exp(
            predicted_log_sigma2 / 2
        )

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
