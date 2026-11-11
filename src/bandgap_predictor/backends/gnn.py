"""HSE06-like GNN backend with a frozen direct/indirect classification head."""

import os
from importlib.metadata import version

from ..errors import PreflightError, RunError
from ..schemas import (
    BackendId,
    ElectronicClassification,
    GapType,
    PredictionValue,
)
from .base import BackendOutcome, PredictionContext


class GNNHSE06Backend:
    backend_id = BackendId.GNN_HSE06
    method = (
        "HSE06-like ScalarPropertyMACE with frozen direct/indirect head "
        "(20250421 backbone low-LR fine-tuned to the InSb VASP HSE06+SOC "
        "(aexx-tuned) reference 0.241 eV; head trained on "
        "semiconductor-only labels)"
    )
    warning = (
        "Bundled GNN model is a trusted-source legacy/provisional PyTorch pickle."
    )

    def predict(self, context: PredictionContext) -> BackendOutcome:
        model = context.assets.verify("gnn-hse06-model")
        feature_map = context.assets.verify("gnn-feature-map")
        try:
            import torch

            def _import_legacy_stack():
                from gnnbgp.macegap.data.atomic_data import AtomicData
                from gnnbgp.macegap.data.utils import config_from_atoms
                from gnnbgp.macegap.tools.predict_property import (
                    ScalarPropertyPredictor,
                )

                return AtomicData, config_from_atoms, ScalarPropertyPredictor

            # e3nn 0.4.4 stores slice objects in its bundled Wigner constants.
            # Torch 2.6+ defaults torch.load to weights-only mode, so allow only
            # that required built-in while importing the pinned legacy stack.
            # Older torch has no safe_globals gate and needs none.
            gate = getattr(torch.serialization, "safe_globals", None)
            if gate is not None:
                with gate([slice]):
                    AtomicData, config_from_atoms, ScalarPropertyPredictor = (
                        _import_legacy_stack()
                    )
            else:
                AtomicData, config_from_atoms, ScalarPropertyPredictor = (
                    _import_legacy_stack()
                )
        except ImportError as error:
            raise PreflightError(
                "GNN dependencies are unavailable; install bandgap-predictor[gnn]"
            ) from error

        context.progress("Loading verified legacy/provisional GNN model")
        try:
            device = os.environ.get("BGP_GNN_DEVICE", "cpu")
            if device not in ("cpu", "cuda"):
                raise PreflightError(f"unsupported BGP_GNN_DEVICE: {device}")
            predictor = ScalarPropertyPredictor(
                str(model.path),
                device=device,
                feat_map_file=str(feature_map.path),
            )
            configuration = config_from_atoms(
                context.structure.atoms,
                scalar_property_key=None,
            )
            atomic_data = AtomicData.from_config(
                configuration,
                z_table=predictor.z_table,
                cutoff=predictor.r_max,
                extra_node_attrs_map=predictor.extra_node_attrs_map,
                knn=24,
            )
            values, _references = predictor.predict(
                [atomic_data],
                include_ref=False,
                batch_size=1,
                knn=24,
            )
            raw_gap = float(values[0])
        except Exception as error:
            raise RunError(f"legacy GNN inference failed: {error}") from error

        # The bundled model carries a two-class direct/indirect head on the
        # frozen scalar backbone.  Its gap output is bit-for-bit the mid-term
        # model's; only the type is new.  If a future model without a head is
        # installed, keep serving the scalar prediction and report no type.
        gap_type = None
        gap_type_probability = None
        class_probabilities = None
        try:
            typed = predictor.predict_band_gap_type(
                [atomic_data],
                include_ref=False,
                knn=24,
                batch_size=1,
            )
            label = typed["predictions"][0]
            gap_type = GapType(label)
            # Softmax columns follow BAND_GAP_TYPE_TO_INDEX: 0=indirect, 1=direct.
            class_probabilities = dict(
                zip(("indirect", "direct"), typed["probabilities"][0])
            )
            gap_type_probability = class_probabilities[label]
        except RuntimeError:
            context.progress("Model has no direct/indirect head; type omitted")

        gap = max(0.0, raw_gap)
        classification = (
            ElectronicClassification.INSULATOR
            if gap > 1e-8
            else ElectronicClassification.ZERO_GAP
        )
        diagnostics = {"raw_prediction_ev": raw_gap}
        if class_probabilities is not None:
            diagnostics["gap_type_class_probabilities"] = class_probabilities
        return BackendOutcome(
            prediction=PredictionValue(
                gap_ev=gap,
                classification=classification,
                gap_type=gap_type,
                gap_type_probability=gap_type_probability,
            ),
            converged=None,
            occupation_summary={},
            settings={"batch_size": 1, "device": device, "knn": 24},
            provenance={
                "feature_map_sha256": feature_map.sha256,
                "mace_torch_version": version("mace-torch"),
                "model_sha256": model.sha256,
                "trust": "legacy-provisional-pickle",
            },
            diagnostics=diagnostics,
            warnings=[self.warning],
        )
