from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list

def get_neighborhood_knn(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    knn: int,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    temp_cell = np.copy(cell)
    if not pbc_x:
        temp_cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        temp_cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        temp_cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, abs_dist, unit_shifts = neighbour_list(
        quantities="ijdS",
        pbc=pbc,
        cell=temp_cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )
    # Trim down the list to k-nearest neighbours
    unique_senders = np.unique(sender)
    mask = np.zeros(len(sender), dtype=bool)
    for i in unique_senders:
        indices = np.where(sender == i)[0]
        # Limit to k-nearest neighbours
        dists_args = indices[np.argsort(abs_dist[indices])[:knn]]
        mask[dists_args] = True
    # Apply masks
    sender = sender[mask]
    receiver = receiver[mask]
    unit_shifts = unit_shifts[mask]
        

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell
