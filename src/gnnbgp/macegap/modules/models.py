
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
#from torch_scatter import scatter_max, scatter_min

from mace.tools.scatter import scatter_sum, scatter_mean, scatter_std

from mace.modules.blocks import (
    LinearNodeEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules import MACE

from mace.modules.utils import (
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

from .blocks import LegacyNonLinearReadoutBlock

class TakeDifference(torch.nn.Module):
    """Take the difference between two modules' outputs.""" 
    def __init__(self, modules, take_abs=False):
        self.modules = modules
        self.take_abs = take_abs

    def forward(self, x):
        output = [module(x) for module in self.modules]
        if self.take_abs is True:
            return abs(output[0] - output[1])
        return output[0] - output[1]




@compile_mode("script")
class ScalarPropertyMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        num_extra_attrs: Optional[int] = 0,
        property_inter_scale: float=1.0,
        property_inter_shift: float=0.,
        global_pool_operation: Optional[str] = None,
        intensive: bool = True,
        classification_num_classes: int = 0,
        classification_hidden_dim: int = 0,
        classification_pool_operation: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The epoch-296 checkpoint used a three-hidden-layer MLP readout at
        # every interaction.  MACE 0.3.14 shortened this block to one hidden
        # layer, so restore the historical layout before cloning
        # ``readouts_p``.  This keeps the original scalar-property heads
        # loadable while leaving the new classifier randomly initialized.
        if len(self.readouts) > 0:
            legacy_readouts = []
            for index in range(len(self.readouts)):
                irreps_in = kwargs["hidden_irreps"]
                if index == len(self.readouts) - 1:
                    irreps_in = o3.Irreps(str(kwargs["hidden_irreps"][0]))
                legacy_readouts.append(
                    LegacyNonLinearReadoutBlock(
                        irreps_in,
                        (len(self.heads) * kwargs["MLP_irreps"]).simplify(),
                        kwargs["gate"],
                        o3.Irreps(f"{len(self.heads)}x0e"),
                        len(self.heads),
                        kwargs.get("cueq_config"),
                        kwargs.get("oeq_config"),
                    )
                )
            self.readouts = torch.nn.ModuleList(legacy_readouts)
        self.readouts_p = deepcopy(self.readouts)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        self.scale_shift_p = ScaleShiftBlock(
            scale=property_inter_scale, shift=property_inter_shift
        )
        self.global_pool_operation : Union[str, None] = global_pool_operation
        self.intensive : bool = intensive
        # Optional graph-level classifier.  It is disabled by default so
        # serialized scalar-property models remain usable unchanged.
        self.classification_num_classes: int = int(classification_num_classes)
        if self.classification_num_classes not in (0, 2):
            raise ValueError(
                "Direct/indirect band-gap classification requires exactly 2 classes"
            )
        self.classification_pool_operation: str = classification_pool_operation
        hidden_irreps = kwargs.get("hidden_irreps")
        if hidden_irreps is None:
            raise ValueError("hidden_irreps is required for the classification head")
        # The classifier receives the concatenation of the feature tensors
        # emitted by every interaction/product block (see ``forward``), not
        # the scalar property readout widths.  Those feature widths can differ
        # between interactions, so derive the input size from the products.
        classifier_input_dim = sum(
            int(product.symmetric_contractions.irreps_out.dim)
            for product in self.products
            if hasattr(product, "symmetric_contractions")
        )
        if classifier_input_dim == 0:
            classifier_input_dim = int(hidden_irreps.dim) * len(self.interactions)
        hidden_dim = int(classification_hidden_dim) or max(32, classifier_input_dim // 2)
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(classifier_input_dim, hidden_dim),
            torch.nn.SiLU(),
            # Keep one output neuron for scalar-only models so the module
            # type is stable for TorchScript; the head is ignored in that mode.
            torch.nn.Linear(hidden_dim, max(1, self.classification_num_classes)),
        )
        # Swap the node embedding layer with the new dimensions
        if num_extra_attrs > 0:
            irreps_in = o3.Irreps([(num_extra_attrs, (0, 1))])
            irreps_out = o3.Irreps([(kwargs.get('hidden_irreps').count(o3.Irrep(0, 1)), (0, 1))])
            self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=irreps_in, irreps_out=irreps_out, 
        )
        
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        graph_heads = (
            data["head"]
            if "head" in data
            else torch.zeros(num_graphs, type=torch.long)
        )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, num_heads]

        
        # Embeddings
        if 'node_extra_attrs' in data:
            node_feats = self.node_embedding(data["node_extra_attrs"])
            data["node_extra_attrs"].requires_grad_(True)
        else:
            node_feats = self.node_embedding(data["node_attrs"])
            
            
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        # Recent MACE releases return ``(features, auxiliary)`` from radial
        # embeddings while older checkpoints expect the feature tensor alone.
        if isinstance(edge_feats, tuple):
            edge_feats = edge_feats[0]
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)
        # Interactions
        node_es_list = [pair_node_energy]
        node_ps_list = []
        global_ps_list = []
        node_feats_list = []
        for interaction, product, readout, readout_p in zip(
            self.interactions, self.products, self.readouts,
            self.readouts_p,
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(
                readout(node_feats, node_heads)[num_atoms_arange, node_heads]
            )  # {[n_nodes, ], }

            if self.global_pool_operation is None:
                # Per node property prediction
                node_ps_list.append(
                    readout_p(node_feats, node_heads)[num_atoms_arange, node_heads]
                )  # {[n_nodes, ], }
            else:
                # Pool the node features into a global feature
                pooled_feat = self.global_pool(node_feats, data, self.global_pool_operation)
                # Read the global (graph) feature 
                global_ps_list.append(
                    readout_p(pooled_feat, graph_heads)
                )

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]

        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Process the property part
        if self.global_pool_operation is None:

            # Sum over properties 
            node_inter_ps = torch.sum(
                torch.stack(node_ps_list, dim=0), dim=0
            )  # [n_nodes, ]

            node_inter_ps = self.scale_shift_p(node_inter_ps, node_heads)

            # Compute property predictions
            if self.intensive:
                total_p = scatter_mean(
                    src=node_inter_ps, index=data["batch"], dim=-1, dim_size=num_graphs
                )
            else:
                total_p = scatter_sum(
                    src=node_inter_ps, index=data["batch"], dim=-1, dim_size=num_graphs
                )
        else:
            total_p = torch.sum(torch.concat(global_ps_list, dim=-1), dim=-1)
            #total_p = self.scale_shift_p(total_p, torch.zeros(num_graphs, dtype=torch.long))
            
                

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "displacement": displacement,
            "node_feats": node_feats_out,
            "scalar_property": total_p,
        }

        # ``hasattr`` is intentional: models serialized before the classifier
        # was introduced do not contain these attributes.
        if hasattr(self, "classification_num_classes") and self.classification_num_classes > 0:
            pooled_classification = self.global_pool(
                node_feats_out, data, self.classification_pool_operation
            )
            output["band_gap_type_logits"] = self.classification_head(
                pooled_classification
            )

        return output




    def global_pool(self, node_feats, data, pool_operation='mean'):
        """Pool the node features into global features"""
        if pool_operation =='mean':
            return scatter_mean(node_feats, data['batch'], dim=0)
        elif pool_operation =='sum':
            return scatter_sum(node_feats, data['batch'], dim=0)
        elif pool_operation == 'std':
            return scatter_std(node_feats, data['batch'], dim=0)
        else:
            raise ValueError(f"Unsupported pool operation: {pool_operation}")

def generate_irreps_from_rank(rank):
    if rank == 0:
        return None, o3.Irreps("1x0e")
    elif rank == 1:
        return 'i', o3.Irreps("1x0e + 1x1o")
    if rank == 2:
        return 'ij', o3.Irreps("1x0e + 1x1o+ 1x2e")
    elif rank == 3:
        return 'ijk', o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
    else:
        raise ValueError(f"Unsupported rank: {rank}")
