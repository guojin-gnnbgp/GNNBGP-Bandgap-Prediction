
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    Linear,
)
from mace.tools.compile import simplify_if_compile


class LegacyNonLinearReadoutBlock(torch.nn.Module):
    """Three-hidden-layer scalar readout used by the epoch-296 checkpoint.

    The current MACE release shortened ``NonLinearReadoutBlock`` to one hidden
    layer.  Keeping the historical ``layers.*`` layout allows the original
    regression heads to be transferred exactly during warm-start training.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config=None,
    ):
        super().__init__()
        del num_heads, oeq_config
        layers = [
            o3.Linear(irreps_in=irreps_in, irreps_out=MLP_irreps, biases=True),
            nn.Activation(irreps_in=MLP_irreps, acts=[gate]),
        ]
        for _ in range(2):
            layers.extend(
                [
                    o3.Linear(irreps_in=MLP_irreps, irreps_out=MLP_irreps, biases=True),
                    nn.Activation(irreps_in=MLP_irreps, acts=[gate]),
                ]
            )
        layers.append(
            o3.Linear(irreps_in=MLP_irreps, irreps_out=irrep_out, biases=True)
        )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, heads: Optional[torch.Tensor] = None) -> torch.Tensor:
        del heads
        for layer in self.layers:
            x = layer(x)
        return x


@compile_mode("script")
class NoneLinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, gate: Optional[Callable]):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.non_linearity = nn.Activation(irreps_in=irreps_out, acts=[gate])

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.non_linearity(self.linear(node_attrs))


@simplify_if_compile
@compile_mode("script")
class NonLinearTensorReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate] + [None] * (len(self.hidden_irreps) -1))
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, cueq_config=cueq_config
        )
        
    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, len(heads)]


