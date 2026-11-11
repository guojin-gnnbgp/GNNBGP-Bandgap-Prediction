from typing import List, Tuple, Union
from pathlib import Path
import logging

import torch
from ase import Atoms
from monty.serialization import loadfn

from ..data.utils import (
    BAND_GAP_INDEX_TO_TYPE,
    load_from_xyz,
)
from ..data.atomic_data import AtomicData
from mace.data import Configuration
from mace.tools import torch_geometric, utils

logger = logging.getLogger(__name__)

class ScalarPropertyPredictor:
    """
    Use trained models to make predictions
    """

    energy_key = "REF_energy"
    forces_key = "REF_forces"
    stress_key = "REF_stress"

    def __init__(self, model_file: str, device: str = "cpu", feat_map_file=None):
        """
        Instantiate a predictor

        :param model_file: Path to the trained model
        :param device: Name of the device
        :param feat_map_file: Name containing extra node attribute mapping
        """

        # This trusted legacy pickle contains nested e3nn TorchScript buffers
        # serialized on CUDA. The outer torch.load map_location is not forwarded
        # to those nested loads, so temporarily make their CPU mapping explicit.
        original_jit_load = torch.jit.load

        def cpu_jit_load(*args, **kwargs):
            kwargs["map_location"] = device
            return original_jit_load(*args, **kwargs)

        torch.jit.load = cpu_jit_load
        try:
            try:
                self.model = torch.load(
                    model_file,
                    weights_only=False,
                    map_location=device,
                )
            except AttributeError:
                import mace.modules.models

                from ..modules.models import ScalarPropertyMACE

                mace.modules.models.ScalarPropertyMACE = ScalarPropertyMACE
                logger.info("Loading model trained with older versions")
                self.model = torch.load(
                    model_file,
                    weights_only=False,
                    map_location=device,
                )
        finally:
            torch.jit.load = original_jit_load

        # Transfer to the device
        self.model.to(device)

        self.device = device
        # The combined direct/indirect model stores the original scalar model
        # under ``gap_model``.  Keep the predictor compatible with both the
        # original MACE object and that wrapper.
        self._gap_model = getattr(self.model, "gap_model", self.model)
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self._gap_model.atomic_numbers]
        )
        self.r_max = float(self._gap_model.r_max)
        # Mapping for element-wise extra node feature
        if feat_map_file is not None:
            self.extra_node_attrs_map = loadfn(feat_map_file)
        else:
            self.extra_node_attrs_map = None

    def get_data_and_config(
        self, xyz_file, property_key, knn=24, band_gap_type_key=None, **kwargs
    ):
        """
        Load a dataset from a xyz file
        """
        _ = kwargs
        # Click passes ``None`` when --knn is omitted; preserve the training
        # default instead of forwarding None to graph construction.
        if knn is None:
            knn = 24
        _, all_configs = load_from_xyz(
            file_path=xyz_file,
            config_type_weights=None,
            energy_key=self.energy_key,
            forces_key=self.forces_key,
            stress_key=self.stress_key,
            virials_key=None,
            dipole_key=None,
            charges_key=None,
            head_key="head",
            extract_atomic_energies=True,
            keep_isolated_atoms=False,
            head_name="Default",
            scalar_property_key=property_key,
            band_gap_type_key=band_gap_type_key,
        )

        # Create the dataset
        dataset = [
            AtomicData.from_config(
                config,
                z_table=self.z_table,
                cutoff=self.r_max,
                extra_node_attrs_map=self.extra_node_attrs_map,
                knn=knn,
            )
            for config in all_configs
        ]
        return dataset, all_configs

    def predict(
        self,
        dataset: Union[List[Configuration], str, Path],
        property_key=None,
        knn=24,
        batch_size=64,
        include_ref=True,
    ) -> Tuple[list, list]:
        """
        Make prediction with a given dataset
        :param include_ref: Also include reference data
        :returns: A list of predicted properties.
        """
        if isinstance(dataset, (str, Path)):
            dataset, _ = self.get_data_and_config(
                dataset, property_key=property_key, knn=knn
            )

        loader = torch_geometric.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
        )
        pred = []
        ref = []
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if include_ref:
                    ref.extend(batch.scalar_property.detach().numpy())
                batch.to(self.device)
                output = self._forward(batch)
                pred.extend(output["scalar_property"].cpu().detach().numpy())
        if not include_ref:
            ref = [None] * len(pred)
        return pred, ref

    def predict_band_gap_type(
        self,
        dataset: Union[List[Configuration], str, Path],
        gap_type_key: str = "band_gap_type",
        knn: int = 24,
        batch_size: int = 64,
        include_ref: bool = True,
    ) -> dict:
        """Predict whether a structure has a direct or indirect gap.

        The model must have been trained with ``classification_num_classes=2``.
        The return value contains human-readable labels, integer class indices,
        and per-class probabilities.  Missing reference labels are returned as
        ``None`` so inference files do not need labels.
        """
        num_classes = int(getattr(self.model, "classification_num_classes", 0) or 0)
        # ``FrozenGapClassifier`` predates the generic model attribute and
        # exposes its two-class head directly.
        if num_classes < 2 and hasattr(self.model, "classification_head"):
            head = self.model.classification_head
            final = next(
                (layer for layer in reversed(list(head.modules())) if isinstance(layer, torch.nn.Linear)),
                None,
            )
            num_classes = int(final.out_features) if final is not None else 0
        if num_classes < 2:
            raise RuntimeError(
                "The loaded model has no direct/indirect classifier. "
                "Train a model with classification_num_classes=2 first."
            )
        if isinstance(dataset, (str, Path)):
            dataset, _ = self.get_data_and_config(
                dataset,
                property_key=None,
                knn=knn,
                band_gap_type_key=gap_type_key,
            )

        loader = torch_geometric.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
        )
        predicted_indices = []
        probabilities = []
        references = []
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.training = False
        with torch.no_grad():
            for batch in loader:
                if include_ref:
                    labels = getattr(batch, "band_gap_type", None)
                    if labels is None:
                        references.extend([None] * int(batch.num_graphs))
                    else:
                        references.extend(
                            [
                                BAND_GAP_INDEX_TO_TYPE.get(int(value), None)
                                for value in labels.detach().cpu().reshape(-1).tolist()
                            ]
                        )
                batch = batch.to(self.device)
                output = self._forward(batch)
                logits = output.get("band_gap_type_logits")
                if logits is None:
                    raise RuntimeError(
                        "The loaded model did not return band_gap_type_logits"
                    )
                probs = torch.softmax(logits, dim=-1).cpu()
                indices = torch.argmax(probs, dim=-1)
                predicted_indices.extend(indices.tolist())
                probabilities.extend(probs.tolist())
        if not include_ref:
            references = [None] * len(predicted_indices)
        predictions = [BAND_GAP_INDEX_TO_TYPE[int(i)] for i in predicted_indices]
        return {
            "predictions": predictions,
            "indices": predicted_indices,
            "probabilities": probabilities,
            "references": references,
        }

    def _forward(self, batch):
        """Run either a legacy scalar model or the combined model."""
        if hasattr(self.model, "gap_model") and hasattr(
            self.model, "classification_head"
        ):
            return self.model.forward(batch.to_dict())
        return self.model.forward(batch.to_dict(), compute_force=False)

    # Short alias for callers that use the generic property naming.
    predict_gap_type = predict_band_gap_type

    @staticmethod
    def config2atoms(config):
        """
        Convert configurations to atoms
        """
        atoms = Atoms(
            numbers=config.atomic_numbers,
            cell=config.cell,
            positions=config.positions,
            pbc=True,
        )
        atoms.info["scalar_property"] = config.scalar_property
        if getattr(config, "band_gap_type", None) in BAND_GAP_INDEX_TO_TYPE:
            atoms.info["band_gap_type"] = BAND_GAP_INDEX_TO_TYPE[config.band_gap_type]
        return atoms

    def get_dataframe(self, dataset, configs=None, **kwargs):
        """Get a pandas dataframe"""
        import pandas as pd

        if isinstance(dataset, str):
            dataset, configs = self.get_data_and_config(dataset, **kwargs)
        pred, target = self.predict(dataset, **kwargs)
        df = pd.DataFrame.from_dict(
            {
                "atoms": [self.config2atoms(config) for config in configs],
                "predict": pred,
                "target": target,
            }
        )
        df["formula"] = df.atoms.apply(lambda x: x.get_chemical_formula())
        return df
