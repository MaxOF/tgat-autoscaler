from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Body
from starlette.concurrency import run_in_threadpool
import asyncio

_operation_lock = asyncio.Lock()


def _run_async(operation):
    return asyncio.run(operation)


async def _service_operation(operation):
    async with _operation_lock:
        return await run_in_threadpool(_run_async, operation)


from app.core.interfaces import AblateRequest, SensitivityRequest, GraphWindow
from app.dto.train_csv_request_dto import TrainCSVRequest
from app.service.tgat_service import TGATService
from config.settings import CONFIG, EDGES_CSV_PATH, NODES_CSV_PATH

router = APIRouter(prefix="/api", tags=["TGAT-Autoscaler"])

tgat_service = TGATService()


# =============================================================================
# HEALTH / CONFIGURATION
# =============================================================================


@router.get("/health")
async def health() -> Dict[str, Any]:
    """
    Lightweight service-health endpoint.

    It intentionally does not query Prometheus or Kubernetes.
    """
    return {
        "status": "ok",
        "service": "TGAT-Autoscaler",
        "model": {
            "time_encoding": tgat_service.model.time_encoding_enabled,
            "hidden_edges": tgat_service.model.use_hidden_edges,
            "graph_enabled": tgat_service.model.use_graph,
            "dropedge_prob": tgat_service.model.dropedge_prob,
        },
        "control": {
            "decision_interval_sec": CONFIG.get("metrics_interval", 300),
            "forecast_horizon_min": CONFIG.get("forecast_horizon_minutes", 5),
            "history_minutes": CONFIG.get("observation_window_minutes", 60),
        },
    }


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Return experiment-relevant runtime configuration.

    Passwords/tokens are not stored in CONFIG, therefore this endpoint
    can be used to record experiment settings for reproducibility.
    """

    return {
        "model": tgat_service.model.ablation_state(),
        "hidden_edge": CONFIG.get("hidden_edge", {}),
        "objective": CONFIG.get("objective", {}),
        "safety": CONFIG.get("safety", {}),
        "slo": CONFIG.get("slo", {}),
        "resource_cost": CONFIG.get("resource_cost", {}),
        "observation_window_minutes": CONFIG.get("observation_window_minutes", 60),
        "forecast_horizon_minutes": CONFIG.get("forecast_horizon_minutes", 5),
        "decision_interval_sec": CONFIG.get("metrics_interval", 300),
    }


# =============================================================================
# TRAINING
# =============================================================================


@router.post("/train")
async def train() -> Dict[str, Any]:
    """
    Train the seven-target TGAT model using the configured CSV dataset.
    """

    try:
        training_request = TrainCSVRequest(
            nodes_csv_path=(NODES_CSV_PATH), edges_csv_path=(EDGES_CSV_PATH)
        )

        return await _service_operation(tgat_service.train_from_csv(training_request))

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=("Training failed: " f"{exc}")
        ) from exc


@router.post("/train-csv")
async def train_csv(request: TrainCSVRequest) -> Dict[str, Any]:
    """
    Train using an explicitly supplied nodes.csv / edges.csv pair.

    This endpoint is useful for:
    - repeated-seed experiments;
    - sensitivity studies;
    - alternative graph datasets;
    - partially observed graph experiments.
    """

    try:
        return await _service_operation(tgat_service.train_from_csv(request))

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=("CSV training failed: " f"{exc}")
        ) from exc


# =============================================================================
# PREDICTION
# =============================================================================


@router.post("/predict")
async def predict(graph: Optional[GraphWindow] = Body(None)) -> Dict[str, Any]:
    """
    Execute telemetry collection and neural prediction only.

    No scaling action is applied to Kubernetes.
    """

    try:
        return await _service_operation(tgat_service.predict(graph))

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=("Prediction failed: " f"{exc}")
        ) from exc


# =============================================================================
# COMPLETE CONTROL LOOP
# =============================================================================


@router.post("/apply")
async def apply(graph: Optional[GraphWindow] = Body(None)) -> Dict[str, Any]:
    """
    Run the complete pipeline:

        telemetry
            -> temporal graph
            -> TGAT prediction
            -> hidden-edge synthesis
            -> multi-objective decision
            -> safety layer
            -> Kubernetes actuation

    Whether Kubernetes is actually mutated depends on safety.dry_run.
    """

    try:
        return await _service_operation(tgat_service.apply(graph))

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=("Autoscaling apply " f"failed: {exc}")
        ) from exc


# =============================================================================
# ABLATION EXPERIMENTS
# =============================================================================


@router.post("/ablate")
async def ablate(request: AblateRequest) -> Dict[str, Any]:
    """
    Configure reviewer-requested ablation experiments.

    Examples
    --------

    Full model:
        {
          "time_encoding": true,
          "hidden_edges": true,
          "graph_enabled": true
        }

    w/o temporal encoding:
        {
          "time_encoding": false
        }

    w/o hidden-edge synthesis:
        {
          "hidden_edges": false
        }

    TGAT-noGraph:
        {
          "graph_enabled": false
        }

    Reduced history:
        {
          "history_minutes": 15
        }
    """
    async with _operation_lock:

        try:
            return tgat_service.ablate(
                time_encoding=(request.time_encoding),
                hidden_edges=(request.hidden_edges),
                graph_enabled=(request.graph_enabled),
                dropedge_prob=(request.dropedge_prob),
                history_minutes=(request.history_minutes),
                safety_enabled=request.safety_enabled,
                hysteresis_enabled=request.hysteresis_enabled,
            )

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=("Ablation configuration " f"failed: {exc}")
            ) from exc


@router.post("/ablate/reset")
async def reset_ablation() -> Dict[str, Any]:
    """
    Restore the default full TGAT configuration.
    """
    async with _operation_lock:

        try:
            return tgat_service.ablate(
                time_encoding=True,
                hidden_edges=True,
                graph_enabled=True,
                dropedge_prob=0.0,
                safety_enabled=True,
                hysteresis_enabled=True,
                history_minutes=(CONFIG.get("observation_window_minutes", 60)),
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=("Ablation reset failed: " f"{exc}")
            ) from exc


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================


@router.post("/sensitivity/config")
async def sensitivity_config(request: SensitivityRequest) -> Dict[str, Any]:
    """
    Change parameters used in reviewer-requested sensitivity analysis.

    This endpoint modifies configuration only. It does not automatically
    retrain the model or execute an experiment.
    """
    async with _operation_lock:

        import copy
        import math

        config_before = copy.deepcopy(CONFIG)
        cfg_before = copy.deepcopy(tgat_service.model.cfg)
        try:
            if request.beta is not None:
                raise ValueError(
                    "beta attention bias is not supported by the GATv2 implementation"
                )
            # ---------------------------------------------------------------------
            # History window W
            # ---------------------------------------------------------------------
            if request.history_minutes is not None:
                if request.history_minutes <= 0:
                    raise ValueError("history_minutes " "must be > 0")

                CONFIG["observation_window_minutes"] = int(request.history_minutes)

                CONFIG["edge_feature_window_sec"] = int(request.history_minutes * 60)

            # ---------------------------------------------------------------------
            # Hidden-edge threshold
            # ---------------------------------------------------------------------
            if request.hidden_edge_threshold is not None:
                threshold = float(request.hidden_edge_threshold)

                if not (0.0 <= threshold <= 1.0):
                    raise ValueError("hidden_edge_threshold " "must be in [0, 1]")

                CONFIG["hidden_edge"]["threshold"] = threshold

            # ---------------------------------------------------------------------
            # Edge-intensity coefficient beta
            #
            # beta is stored in the runtime model configuration. The current
            # compact implementation does not yet expose it in GATv2Conv itself,
            # but retaining the parameter here makes experiment configuration
            # explicit and allows the temporal-attention implementation to use
            # the same value once the custom attention layer is enabled.
            # ---------------------------------------------------------------------
            if request.beta is not None:
                beta = float(request.beta)

                if beta < 0.0:
                    raise ValueError("beta must be >= 0")

                tgat_service.model.cfg["beta"] = beta

            # ---------------------------------------------------------------------
            # Multi-objective weights
            # ---------------------------------------------------------------------
            objective = CONFIG["objective"]

            supplied_weights = {
                "lambda_slo": request.lambda_slo,
                "lambda_cost": request.lambda_cost,
                "lambda_stability": request.lambda_stability,
                "lambda_risk": request.lambda_risk,
            }

            changed = False

            for key, value in supplied_weights.items():
                if value is None:
                    continue

                value = float(value)

                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{key} must be >= 0")

                objective[key] = value

                changed = True

            if changed:
                total = sum(
                    float(objective[key])
                    for key in (
                        "lambda_slo",
                        "lambda_cost",
                        "lambda_stability",
                        "lambda_risk",
                    )
                )

                if total <= 0.0:
                    raise ValueError("At least one objective " "weight must be > 0")

                # Normalize so the controller always satisfies:
                #
                # sum(lambda_k) = 1.
                for key in (
                    "lambda_slo",
                    "lambda_cost",
                    "lambda_stability",
                    "lambda_risk",
                ):
                    objective[key] = float(objective[key]) / total

            return {
                "status": "ok",
                "observation_window_minutes": CONFIG.get(
                    "observation_window_minutes", 60
                ),
                "hidden_edge": CONFIG.get("hidden_edge", {}),
                "beta": tgat_service.model.cfg.get("beta", 0.0),
                "objective": CONFIG.get("objective", {}),
            }

        except ValueError as exc:
            CONFIG.clear()
            CONFIG.update(config_before)
            tgat_service.model.cfg = cfg_before
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=("Sensitivity configuration " f"failed: {exc}")
            ) from exc


# =============================================================================
# HIDDEN-EDGE DIAGNOSTICS
# =============================================================================


@router.get("/hidden-edges")
async def hidden_edges() -> Dict[str, Any]:
    """
    Return reconstructed dependencies accepted during the most recent
    inference pass.

    These records can be compared against a ground-truth topology to
    calculate precision, recall, and F1.
    """

    try:
        edges = tgat_service.model.get_last_hidden_edges()

        return {"count": len(edges), "edges": edges}

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=("Cannot return " f"hidden edges: {exc}")
        ) from exc
