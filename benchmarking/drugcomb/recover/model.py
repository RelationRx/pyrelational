"""For this task we choose the model proposed in

    Bertin, P., Rector-Brooks, J., Sharma, D., Gaudelet, T., Anighoro, A., Gross, T.,
    Martínez-Peña, F., Tang, E.L., Suraj, M.S., Regep, C. and Hayter, J.B., et al. 2023.
    RECOVER identifies synergistic drug combinations in vitro through sequential model optimization.
    Cell Reports Methods, 3(10).
"""

from typing import List

import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter


class BilinearFiLMMLPPredictor(Module):

    def __init__(
        self, drugs_dim: int, cell_lines_dim: int, encoder_layer_dims: List[int], decoder_layer_dims: List[int]
    ):
        """Initialise Recover model.

        :param drugs_dim: dimension of drugs' features
        :param cell_lines_dim: dimension of cell lines' features
        :param encoder_layer_dims: list of sequential dims for encoder layers, should not contain drug embeddings
        :param decoder_layer_dims: list of sequential dims for decoder layers
        """
        super().__init__()
        encoder_layer_dims.insert(0, drugs_dim)
        self.encoder = MLPFiLMModule(encoder_layer_dims, cell_lines_dim)
        self.decoder = MLPFiLMModule(decoder_layer_dims, cell_lines_dim)

        merge_dim = encoder_layer_dims[-1]
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((merge_dim, merge_dim, merge_dim))
            + torch.cat([torch.eye(merge_dim)[None, :, :]] * merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((merge_dim)))
        self.bilinear_diag = Parameter(1 / 100 * torch.randn((merge_dim, merge_dim)) + 1)

    def forward(self, drugs_a: Tensor, drugs_b: Tensor, cell_lines: Tensor) -> Tensor:
        drugs_a = self.encoder(drugs_a, cell_lines)
        drugs_b = self.encoder(drugs_b, cell_lines)
        drugs_a = self.bilinear_weights.matmul(drugs_a.T).T
        drugs_b = self.bilinear_weights.matmul(drugs_b.T).T * self.bilinear_diag
        drugs_a = drugs_a.permute(0, 2, 1)
        combos = (drugs_a * drugs_b).sum(1) + self.bilinear_offsets
        return self.decoder.forward(combos, cell_lines)


class MLPFiLMModule(Module):

    def __init__(self, layer_dims: List[int], cell_lines_dim: int):
        super().__init__()
        self.mlp = torch.nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.mlp.append(
                LinearFiLMModule(layer_dims[i], layer_dims[i + 1], cell_lines_dim),
            )

    def forward(self, drug: Tensor, cell_lines: Tensor) -> Tensor:
        for i, layer in enumerate(self.mlp):
            drug = layer(drug, cell_lines)
            if i < len(self.mlp) - 1:
                drug = torch.relu(drug)
        return drug


class LinearFiLMModule(Module):
    def __init__(self, in_dim: int, out_dim: int, conditioning_dim: int):
        super(LinearFiLMModule, self).__init__()
        self.out_dim = out_dim
        self.linear = Linear(in_dim, out_dim)
        self.conditioning = Linear(conditioning_dim, 2 * out_dim)
        self.conditioning.bias.data[:out_dim] += 1

    def forward(self, drugs: Tensor, cell_lines: Tensor) -> Tensor:
        drugs = self.linear(drugs)
        conditioning = self.conditioning(cell_lines)
        drugs: Tensor = conditioning[:, : self.out_dim] * drugs + conditioning[:, self.out_dim :]
        return drugs
