from typing import Optional, Union

import torch

from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

from .neighborhood import get_neighborhood_knn
from mace.data.neighborhood import get_neighborhood
from .utils import Configuration


class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    node_extra_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor
    scalar_property_value: torch.Tensor
    scalar_property_weight: torch.Tensor
    band_gap_type: torch.Tensor
    band_gap_type_weight: torch.Tensor
    tensor_property_value: torch.Tensor
    tensor_property_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        head: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipole: Optional[torch.Tensor],  # [, 3]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
        scalar_property: Optional[torch.Tensor],  # [, ]
        tensor_property: Optional[torch.Tensor],  # [n_nodes, ... ] Holds per-node tensor properties
        scalar_property_weight: Optional[torch.Tensor],  # [, ] 
        tensor_property_weight: Optional[torch.Tensor],  # [, ] 
        node_extra_attrs: Optional[torch.Tensor],  # [n_nodes, n_node_extra_feats]
        band_gap_type: Optional[torch.Tensor] = None,  # [, ] 0=indirect, 1=direct
        band_gap_type_weight: Optional[torch.Tensor] = None,  # [, ]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert weight is None or len(weight.shape) == 0
        assert head is None or len(head.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert virials_weight is None or len(virials_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipole is None or dipole.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        assert scalar_property is None or len(scalar_property.shape) == 0
        assert scalar_property_weight is None or len(scalar_property_weight.shape) == 0
        assert band_gap_type is None or len(band_gap_type.shape) == 0
        assert band_gap_type_weight is None or len(band_gap_type_weight.shape) == 0
        assert tensor_property is None or len(tensor_property.shape) > 0
        assert tensor_property_weight is None or len(tensor_property_weight.shape) == 0

        # Aggregate data
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "node_extra_attrs": node_extra_attrs,
            "weight": weight,
            "head": head,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipole": dipole,
            "charges": charges,
            "scalar_property": scalar_property,
            "scalar_property_weight": scalar_property_weight,
            "band_gap_type": band_gap_type,
            "band_gap_type_weight": band_gap_type_weight,
            "tensor_property": tensor_property,
            "tensor_property_weight": tensor_property_weight
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls, config: Configuration, z_table: AtomicNumberTable, cutoff: float,
        extra_node_attrs_map: Union[None, dict] = None, heads=None, knn: int = 0
    ) -> "AtomicData":

        if heads is None:
            heads = ["default"]
        if knn is not None and knn > 0:
            edge_index, shifts, unit_shifts, cell = get_neighborhood_knn(
                positions=config.positions,
                cutoff=cutoff,
                pbc=config.pbc,
                cell=config.cell,
                knn=knn,
            )
        else:
            edge_index, shifts, unit_shifts, cell = get_neighborhood(
                positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
            )
                
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        if extra_node_attrs_map:
            # Construct the extra attrs for each node
            node_extra_attrs = torch.stack(
                [torch.tensor(extra_node_attrs_map[str(i)], dtype=torch.get_default_dtype()) for i in config.atomic_numbers],
                dim=0,
            )
        else:
            node_extra_attrs = None

        # Attach additional node attributes, if provided
        if config.additional_node_attrs is not None:
            node_extra_attrs = torch.cat([one_hot, 
                                          torch.tensor(config.additional_node_attrs, 
                                                       dtype=torch.get_default_dtype())], dim=1)
        try:
            head = torch.tensor(heads.index(config.head), dtype=torch.long)
        except ValueError:
            head = torch.tensor(len(heads) - 1, dtype=torch.long)

        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else 1
        )

        energy_weight = (
            torch.tensor(config.energy_weight, dtype=torch.get_default_dtype())
            if config.energy_weight is not None
            else 1
        )

        forces_weight = (
            torch.tensor(config.forces_weight, dtype=torch.get_default_dtype())
            if config.forces_weight is not None
            else 1
        )

        stress_weight = (
            torch.tensor(config.stress_weight, dtype=torch.get_default_dtype())
            if config.stress_weight is not None
            else 1
        )

        virials_weight = (
            torch.tensor(config.virials_weight, dtype=torch.get_default_dtype())
            if config.virials_weight is not None
            else 1
        )

        forces = (
            torch.tensor(config.forces, dtype=torch.get_default_dtype())
            if config.forces is not None
            else None
        )
        energy = (
            torch.tensor(config.energy, dtype=torch.get_default_dtype())
            if config.energy is not None
            else None
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(config.stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.stress is not None
            else None
        )
        virials = (
            voigt_to_matrix(
                torch.tensor(config.virials, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.virials is not None
            else None
        )
        dipole = (
            torch.tensor(config.dipole, dtype=torch.get_default_dtype()).unsqueeze(0)
            if config.dipole is not None
            else None
        )
        charges = (
            torch.tensor(config.charges, dtype=torch.get_default_dtype())
            if config.charges is not None
            else None
        )
        scalar_property = (
            torch.tensor(config.scalar_property, dtype=torch.get_default_dtype())
            if config.scalar_property is not None
            else None
        )
        scalar_property_weight = (
            torch.tensor(config.scalar_property_weight, dtype=torch.get_default_dtype())
            if config.scalar_property_weight is not None
            else 0
        )
        band_gap_type = torch.tensor(
            -1 if config.band_gap_type is None else config.band_gap_type,
            dtype=torch.long,
        )
        band_gap_type_weight = torch.tensor(
            config.band_gap_type_weight, dtype=torch.get_default_dtype()
        )
        tensor_property = (
            torch.tensor(config.tensor_property, dtype=torch.get_default_dtype())
            if config.tensor_property is not None
            else None
        )
        tensor_property_weight = (
            torch.tensor(config.tensor_property_weight, dtype=torch.get_default_dtype())
            if config.tensor_property_weight is not None
            else 0
        )


        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            node_attrs=one_hot,
            node_extra_attrs=node_extra_attrs,
            weight=weight,
            head=head,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
            virials_weight=virials_weight,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipole=dipole,
            charges=charges,
            scalar_property=scalar_property,
            scalar_property_weight=scalar_property_weight,
            band_gap_type=band_gap_type,
            band_gap_type_weight=band_gap_type_weight,
            tensor_property=tensor_property,
            tensor_property_weight=tensor_property_weight
        )
