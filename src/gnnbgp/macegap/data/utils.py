
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ase.data
import ase.io
import numpy as np

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ], [3,3], [9, ]
Virials = np.ndarray  # [6, ], [3,3], [9, ]
Charges = np.ndarray  # [..., 1]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)


# The order is part of the on-disk/training data contract.  Keep it stable so
# models trained by different versions of gnnbgp produce the same labels.
BAND_GAP_TYPE_TO_INDEX = {"indirect": 0, "direct": 1}
BAND_GAP_INDEX_TO_TYPE = {value: key for key, value in BAND_GAP_TYPE_TO_INDEX.items()}


def normalize_band_gap_type(value) -> Optional[int]:
    """Convert common direct/indirect labels to the stable integer encoding.

    ``None`` means that no label was supplied.  Invalid non-null values raise
    ``ValueError`` instead of silently turning a malformed training example
    into a different class.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value in BAND_GAP_INDEX_TO_TYPE:
            return value
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
        if value in BAND_GAP_INDEX_TO_TYPE:
            return value
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if text in {"direct", "direct gap", "d", "1", "true", "yes"}:
        return BAND_GAP_TYPE_TO_INDEX["direct"]
    if text in {"indirect", "indirect gap", "i", "0", "false", "no"}:
        return BAND_GAP_TYPE_TO_INDEX["indirect"]
    raise ValueError(
        f"Unsupported band-gap type {value!r}; expected direct/indirect or 0/1"
    )


def classify_band_gap(band_gap: float, direct_gap: float, tolerance: float = 1e-3) -> str:
    """Classify a calculated gap from fundamental and direct gap values."""
    return "direct" if abs(float(direct_gap) - float(band_gap)) <= tolerance else "indirect"


def get_band_gap_type(info: Dict, key: Optional[str] = None) -> Tuple[Optional[int], float]:
    """Read a direct/indirect label from ASE ``Atoms.info``.

    The explicit key takes precedence.  For convenience, the standard keys
    ``band_gap_type``, ``gap_type`` and ``is_direct`` are also recognized.  A
    pair of ``band_gap``/``direct_gap`` values can be used as a fallback when
    no categorical label is present; equal gaps indicate a direct gap.
    """
    keys = [key] if key else ["band_gap_type", "gap_type", "is_direct"]
    for candidate in keys:
        if candidate and candidate in info:
            return normalize_band_gap_type(info[candidate]), 1.0

    if key is None and "band_gap" in info and "direct_gap" in info:
        try:
            # A small tolerance accommodates rounded values in extxyz files.
            direct = float(info["direct_gap"])
            fundamental = float(info["band_gap"])
            return normalize_band_gap_type(classify_band_gap(fundamental, direct)), 1.0
        except (TypeError, ValueError):
            pass
    return None, 0.0

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipole: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config
    head: Optional[str] = "Default"  # head used to compute the config
    additional_node_attrs: Optional[np.ndarray] = None
    scalar_property: Optional[float] = None
    scalar_property_weight: float = 1.0
    tensor_property: Optional[float] = None # Any additional tensor property
    tensor_property_weight: float = 1.0
    band_gap_type: Optional[int] = None  # 0=indirect, 1=direct
    band_gap_type_weight: float = 1.0


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int, work_dir: str
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if len(indices[train_size:]) < 10:
        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with following indices: {indices[train_size:]}"
        )
    else:
        # Save indices to file
        with open(work_dir + f"/valid_indices_{seed}.txt", "w", encoding="utf-8") as f:
            for index in indices[train_size:]:
                f.write(f"{index}\n")

        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with indices saved in: {work_dir}/valid_indices_{seed}.txt"
        )

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="REF_energy",
    forces_key="REF_forces",
    stress_key="REF_stress",
    virials_key="REF_virials",
    dipole_key="REF_dipole",
    charges_key="REF_charges",
    head_key="head",
    config_type_weights: Optional[Dict[str, float]] = None,
    scalar_property_key: str = None,
    tensor_property_key: str = None,
    tensor_property_shape: Optional[Tuple[int]] = None,
    band_gap_type_key: str = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                head_key=head_key,
                config_type_weights=config_type_weights,
                scalar_property_key=scalar_property_key,
                tensor_property_key=tensor_property_key,
                tensor_property_shape=tensor_property_shape,
                band_gap_type_key=band_gap_type_key,
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="REF_energy",
    forces_key="REF_forces",
    stress_key="REF_stress",
    virials_key="REF_virials",
    dipole_key="REF_dipole",
    charges_key="REF_charges",
    head_key="head",
    config_type_weights: Optional[Dict[str, float]] = None,
    scalar_property_key=None,
    tensor_property_key=None,
    tensor_property_shape=None,
    band_gap_type_key=None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang ^ 3
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)
    scalar_property_weight = atoms.info.get("config_scalar_property_weight", 1.0)
    scalar_property = atoms.info.get(scalar_property_key, None)
    band_gap_type, band_gap_type_weight = get_band_gap_type(
        atoms.info, key=band_gap_type_key
    )
    band_gap_type_weight = atoms.info.get(
        "config_band_gap_type_weight", band_gap_type_weight
    )

    # Load tensor property if it exists and reshape it accordingly
    # This is assumes that previously the tensor property was stored as a flat array for each atom in the 'C' (default) order
    
    if tensor_property_key in atoms.arrays:
        tensor_property = atoms.get_array(tensor_property_key).reshape((len(atoms), *tensor_property_shape))
    else:
        tensor_property = None
    tensor_property_weight = atoms.info.get("config_tensor_property_weight", 1.0)

    head = atoms.info.get(head_key, "Default")

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0
    if dipole is None:
        dipole = np.zeros(3)
        # dipoles_weight = 0.0
    if scalar_property is None:
        scalar_property = 0.0
        scalar_property_weight = 0.0

    if band_gap_type is None:
        # A sentinel keeps PyTorch Geometric batches rectangular while the
        # weight prevents unlabelled structures from contributing to loss.
        band_gap_type = -1
        band_gap_type_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        head=head,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
        scalar_property=scalar_property,
        scalar_property_weight=scalar_property_weight,
        band_gap_type=band_gap_type,
        band_gap_type_weight=band_gap_type_weight,
        additional_node_attrs=atoms.arrays.get("additional_node_attrs", None),
        tensor_property=tensor_property,
        tensor_property_weight=tensor_property_weight,
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        config_type_name = conf.config_type + "_" + conf.head
        if config_type_name not in all_cts:
            all_cts.append(config_type_name)
            test_by_ct.append((config_type_name, [conf]))
        else:
            ind = all_cts.index(config_type_name)
            test_by_ct[ind][1].append(conf)
    return test_by_ct


def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "REF_virials",
    dipole_key: str = "REF_dipole",
    charges_key: str = "REF_charges",
    head_key: str = "head",
    head_name: str = "Default",
    extract_atomic_energies: bool = False,
    keep_isolated_atoms: bool = False,
    scalar_property_key: str = None,
    tensor_property_key: str = None,
    tensor_property_shape = None,
    band_gap_type_key: str = None,
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")
    if energy_key == "energy":
        logging.warning(
            "Since ASE version 3.23.0b1, using energy_key 'energy' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'energy' to 'REF_energy'. You need to use --energy_key='REF_energy' to specify the chosen key name."
        )
        energy_key = "REF_energy"
        for atoms in atoms_list:
            try:
                atoms.info["REF_energy"] = atoms.get_potential_energy()
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract energy: {e}")
                atoms.info["REF_energy"] = None
    if forces_key == "forces":
        logging.warning(
            "Since ASE version 3.23.0b1, using forces_key 'forces' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'forces' to 'REF_forces'. You need to use --forces_key='REF_forces' to specify the chosen key name."
        )
        forces_key = "REF_forces"
        for atoms in atoms_list:
            try:
                atoms.arrays["REF_forces"] = atoms.get_forces()
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract forces: {e}")
                atoms.arrays["REF_forces"] = None
    if stress_key == "stress":
        logging.warning(
            "Since ASE version 3.23.0b1, using stress_key 'stress' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'stress' to 'REF_stress'. You need to use --stress_key='REF_stress' to specify the chosen key name."
        )
        stress_key = "REF_stress"
        for atoms in atoms_list:
            try:
                atoms.info["REF_stress"] = atoms.get_stress()
            except Exception as e:  # pylint: disable=W0703
                atoms.info["REF_stress"] = None
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            isolated_atom_config = (
                len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"
            )
            if isolated_atom_config:
                atomic_number = int(atoms.get_atomic_numbers()[0])
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atomic_number] = float(atoms.info[energy_key])
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy will be used."
                    )
                    atomic_energies_dict[atomic_number] = 0.0
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")
        if not keep_isolated_atoms:
            atoms_list = atoms_without_iso_atoms

    for atoms in atoms_list:
        atoms.info[head_key] = head_name

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        scalar_property_key=scalar_property_key,
        head_key=head_key,
        tensor_property_key=tensor_property_key,
        tensor_property_shape=tensor_property_shape,
        band_gap_type_key=band_gap_type_key,
    )
    return atomic_energies_dict, configs
