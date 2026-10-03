import asyncio
import copy
import importlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from app.core.interfaces import (
    Action,
    GraphWindow,
    NodeFeatures,
    EdgeEvent,
    PolicyConfig,
)
from app.core.model import TGATAutoscalerModel
from app.core.safety import SafetyPolicy
from app.core.state import MemoryStateStore, JSONStateStore
from app.core.actuation import KubernetesActuator
from app.data.csv_dataset import CSVGraphDataset
from app.synthetic.generator import GeneratorConfig, SyntheticDatasetGenerator
from app.dto.train_csv_request_dto import TrainCSVRequest
from app.training.trainer import CSVTrainer
from app.utils.parsers import parse_mem_mib, format_mem_gi_from_mib, parse_iso8601
from config.settings import DEFAULT_MODEL_CFG, FEATURE_ORDER


def model(**kw):
    return TGATAutoscalerModel(
        {**DEFAULT_MODEL_CFG, "d_model": 16, "heads": 2, "layers": 1, **kw}
    )


def node(service="orders-service", level=1.0):
    return NodeFeatures(id=service, x=[level] * len(FEATURE_ORDER))


def graph(events=None):
    return GraphWindow(
        window="2026-08-01T01:00:00Z",
        nodes=[node(), node("products-service", 2)],
        events=events or [],
    )


def event(tau):
    return EdgeEvent(
        src="orders-service", dst="products-service", tau=tau, e=[0.6, 100.0, 0.0]
    )


def dataset(tmp_path, steps=6, empty_edges=False):
    rows = []
    for step in range(steps):
        t = pd.Timestamp("2026-08-01T00:00:00Z") + pd.Timedelta(minutes=5 * step)
        for sid in ["orders-service", "products-service", "payments-service"]:
            rows.append(
                {
                    "window_utc": t.isoformat(),
                    "service": sid,
                    **{k: float(step + 1) for k in FEATURE_ORDER},
                    "target_replicas": step + 2,
                    "target_cpu_m": 500 + step * 100,
                    "target_mem_mib": 512 + step * 10,
                    "error_rate": 0.0,
                }
            )
    nodes = tmp_path / "nodes.csv"
    pd.DataFrame(rows).to_csv(nodes, index=False)
    edges = tmp_path / "edges.csv"
    pd.DataFrame(
        (
            []
            if empty_edges
            else [
                {
                    "window_utc": r["window_utc"],
                    "src": "orders-service",
                    "dst": "products-service",
                    "edge_weight": 0.6,
                }
                for r in rows[::3]
            ]
        ),
        columns=["window_utc", "src", "dst", "edge_weight"],
    ).to_csv(edges, index=False)
    return CSVGraphDataset(nodes, edges), nodes, edges


def test_future_resource_targets(tmp_path):
    ds, _, _ = dataset(tmp_path)
    sample = next(ds.iter_samples())
    assert sample.targets[0, 1:4].tolist() == [3.0, 600.0, 522.0]


def test_empty_observed_graph_preserved(tmp_path):
    ds, _, _ = dataset(tmp_path, empty_edges=True)
    assert next(ds.iter_samples()).graph.events == []


def test_missing_explicit_edge_file_fails(tmp_path):
    _, nodes, _ = dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        CSVGraphDataset(nodes, tmp_path / "missing.csv")


def test_horizon_gap_is_not_next_row(tmp_path):
    ds, nodes, edges = dataset(tmp_path)
    df = pd.read_csv(nodes)
    df = df[~df.window_utc.str.contains("00:05:00")]
    df.to_csv(nodes, index=False)
    samples = list(CSVGraphDataset(nodes, edges).iter_samples())
    assert samples[0].graph.window.startswith("2026-08-01T00:10")


def test_duplicate_csv_rows_rejected(tmp_path):
    _, nodes, edges = dataset(tmp_path)
    df = pd.read_csv(nodes)
    pd.concat([df, df.iloc[:1]]).to_csv(nodes, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        CSVGraphDataset(nodes, edges)


def test_feature_scaler_not_refitted():
    m = model(model_path="does-not-exist.pt")
    m.feature_mu = np.zeros(8, dtype=np.float32)
    m.feature_sd = np.ones(8, dtype=np.float32)
    x, _, _ = m.build_graph_from_payload(graph())
    assert x[0, 0] == 1.0 and x[1, 0] == 2.0
    assert np.all(m.feature_mu == 0)


def test_future_and_stale_events_excluded():
    m = model(model_path="does-not-exist.pt")
    _, ei, _ = m.build_graph_from_payload(
        graph(
            [
                event("2026-08-01T01:05:00Z"),
                event("2026-07-31T23:55:00Z"),
                event("2026-08-01T00:55:00Z"),
            ]
        )
    )
    assert ei.shape == (2, 1)


def test_neighbor_budget(monkeypatch):
    from config.settings import CONFIG

    monkeypatch.setitem(CONFIG, "max_temporal_neighbors", 1)
    _, ei, _ = model(model_path="missing.pt").build_graph_from_payload(
        graph([event("2026-08-01T00:55:00Z"), event("2026-08-01T00:50:00Z")])
    )
    assert ei.shape[1] == 1


def test_duplicate_node_ids():
    gw = graph()
    gw.nodes = [node(), node()]
    with pytest.raises(ValueError, match="Duplicate"):
        model().build_graph_from_payload(gw)


def test_kubernetes_memory_units_and_roundtrip():
    assert parse_mem_mib("1Gi") == 1024.0
    assert parse_mem_mib("1G") == pytest.approx(1e9 / 2**20)
    assert parse_mem_mib("1M") == pytest.approx(1e6 / 2**20)
    assert parse_mem_mib(format_mem_gi_from_mib(512.0)) == 512.0
    with pytest.raises(ValueError):
        parse_mem_mib("garbage")


def test_timezone_required():
    with pytest.raises(ValueError):
        parse_iso8601("2026-08-01T00:00:00")


def policy(**kw):
    return SafetyPolicy(
        PolicyConfig(cooldown_sec=0, smoothing_alpha=1.0, **kw),
        state_store=MemoryStateStore(),
    )


def test_hysteresis_reversals_do_not_accumulate():
    p = policy(hysteresis_windows=2, rate_limit_replicas=5)
    sid = "orders-service"
    p.filter([Action(id=sid, replicas=5)], "2026-08-01T00:00:00Z")
    assert (
        p.filter([Action(id=sid, replicas=7)], "2026-08-01T00:05:00Z")[0].replicas == 5
    )
    assert (
        p.filter([Action(id=sid, replicas=3)], "2026-08-01T00:10:00Z")[0].replicas == 5
    )
    assert (
        p.filter([Action(id=sid, replicas=3)], "2026-08-01T00:15:00Z")[0].replicas == 3
    )


def test_cooldown_uses_logical_clock():
    p = SafetyPolicy(
        PolicyConfig(cooldown_sec=600, hysteresis_windows=1, smoothing_alpha=1.0),
        state_store=MemoryStateStore(),
    )
    sid = "orders-service"
    p.filter([Action(id=sid, replicas=2)], "2026-08-01T00:00:00Z")
    assert (
        p.filter([Action(id=sid, replicas=3)], "2026-08-01T00:05:00Z")[0].replicas == 2
    )
    assert (
        p.filter([Action(id=sid, replicas=3)], "2026-08-01T00:10:00Z")[0].replicas == 3
    )


def test_first_action_rate_limited_from_current():
    p = policy(hysteresis_windows=1, rate_limit_replicas=2)
    result = p.filter(
        [Action(id="orders-service", replicas=20)],
        "2026-08-01T00:00:00Z",
        current_actions=[Action(id="orders-service", replicas=2)],
    )
    assert result[0].replicas == 4


def test_state_other_services_survive():
    p = policy(hysteresis_windows=1)
    p.filter(
        [
            Action(id="orders-service", replicas=2),
            Action(id="products-service", replicas=3),
        ],
        "2026-08-01T00:00:00Z",
    )
    p.filter([Action(id="orders-service", replicas=2)], "2026-08-01T00:05:00Z")
    assert p.state["last_actions"]["products-service"]["replicas"] == 3


def test_json_state_atomic_roundtrip(tmp_path):
    store = JSONStateStore(tmp_path / "state.json")
    store.save({"a": 1})
    store.save({"a": 2})
    assert store.load() == {"a": 2}
    assert len(list(tmp_path.iterdir())) == 1


def test_generator_repeatable_and_mask_independent(tmp_path):
    configs = [
        GeneratorConfig(
            "2026-08-24",
            "2026-08-26",
            str(tmp_path / str(i)),
            seed=4,
            step_minutes=60,
            missing_edge_rate=rate,
        )
        for i, rate in enumerate([0.0, 1.0, 1.0])
    ]
    # Start cycle at day 1, so use a complete 30-day block to reach masking.
    configs = [
        GeneratorConfig(
            "2026-08-01",
            "2026-08-31",
            c.output_dir,
            c.seed,
            c.step_minutes,
            c.missing_edge_rate,
        )
        for c in configs
    ]
    reports = [SyntheticDatasetGenerator(c).generate() for c in configs]
    assert reports[0]["sha256"]["nodes.csv"] == reports[1]["sha256"]["nodes.csv"]
    assert reports[1]["sha256"] == reports[2]["sha256"]
    edges = pd.read_csv(tmp_path / "1" / "edges.csv")
    assert not (edges.scenario == "partial_observability").any()


def test_two_calendar_months_have_exact_windows():
    times = GeneratorConfig("2026-08-01", "2026-10-01").timestamps()
    assert len(times) == 61 * 288
    assert times[-1] == pd.Timestamp("2026-09-30T23:55:00Z")


def test_generator_does_not_overwrite(tmp_path):
    cfg = GeneratorConfig("2026-08-01", "2026-08-02", str(tmp_path), step_minutes=60)
    SyntheticDatasetGenerator(cfg).generate()
    with pytest.raises(FileExistsError):
        SyntheticDatasetGenerator(cfg).generate()


def test_checkpoint_roundtrip_and_train_split(tmp_path):
    ds, nodes, edges = dataset(tmp_path, steps=12)
    m = model()
    path = tmp_path / "model.pt"
    torch.set_num_threads(1)
    result = CSVTrainer(m).train(
        TrainCSVRequest(
            nodes_csv_path=str(nodes),
            edges_csv_path=str(edges),
            epochs=1,
            model_path=str(path),
            device="cpu",
        )
    )
    assert result["training_samples"] + result["validation_samples"] < result["samples"]
    sample = next(ds.iter_samples())
    before = m.predict_targets(*m.build_graph_from_payload(sample.graph))
    loaded = model(model_path=str(path))
    after = loaded.predict_targets(*loaded.build_graph_from_payload(sample.graph))
    np.testing.assert_allclose(
        [[p.rps, p.cpu_milli, p.slo_risk] for p in before],
        [[p.rps, p.cpu_milli, p.slo_risk] for p in after],
        rtol=1e-5,
    )
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model"].pop(next(iter(checkpoint["model"])))
    torch.save(checkpoint, tmp_path / "incomplete.pt")
    assert not model().load_checkpoint(str(tmp_path / "incomplete.pt"))


def test_empty_graph_uses_trained_projection():
    m = model()
    m.init_model(8, 12)
    out = m.model(torch.ones((2, 8)), None, None)
    assert out.shape == (2, 7) and torch.isfinite(out).all()
    out.sum().backward()
    assert m.model.input_projection.weight.grad is not None


def test_actuator_failure_not_reported_success(monkeypatch):
    import app.core.actuation as act

    class FakeAPI:
        def patch_namespaced_deployment_scale(self, **kw):
            raise RuntimeError("denied")

    monkeypatch.setattr(act.config, "load_kube_config", lambda: None)
    monkeypatch.setattr(act.client, "AppsV1Api", lambda: FakeAPI())
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    report = KubernetesActuator(PolicyConfig(dry_run=False)).apply(
        [Action(id="orders-service", replicas=2)]
    )
    assert report["patched"] == [] and report["details"][0]["errors"]


def test_health_without_external_calls():
    from main import create_app

    with TestClient(create_app()) as client:
        assert client.get("/api/health").status_code == 200


def test_duplicate_window_does_not_advance_hysteresis():
    p = policy(hysteresis_windows=2, rate_limit_replicas=5)
    p.filter([Action(id="orders-service", replicas=5)], "2026-08-01T00:00:00Z")
    requested = [Action(id="orders-service", replicas=7)]
    assert p.filter(requested, "2026-08-01T00:05:00Z")[0].replicas == 5
    assert p.filter(requested, "2026-08-01T00:05:00Z")[0].replicas == 5
    assert p.filter(requested, "2026-08-01T00:10:00Z")[0].replicas == 7


def test_slo_objective_depends_on_capacity():
    from app.core.interfaces import Prediction

    p = policy(
        hysteresis_windows=1,
        lambda_slo=1.0,
        lambda_cost=0.0,
        lambda_stability=0.0,
        lambda_risk=0.0,
    )
    prediction = Prediction(
        id="orders-service",
        rps=200,
        replicas=5,
        cpu_milli=800,
        mem_mib=512,
        p95_ms=800,
        p99_ms=1000,
        slo_risk=0.9,
    )
    low = Action(id=prediction.id, replicas=1, cpu="800m", mem="512Mi")
    sufficient = Action(id=prediction.id, replicas=5, cpu="800m", mem="512Mi")
    assert p.objective(prediction, low) > p.objective(prediction, sufficient)


def test_missing_prometheus_telemetry_not_zero_load():
    from app.core.graph import _query_service_metric

    fake = SimpleNamespace(
        candidate_queries={"rps_in": ["q"]},
        serivce_label_keys=["service"],
        query=lambda _: {"data": {"result": []}},
    )
    assert _query_service_metric(fake, "rps_in", "orders-service") is None
    fake.query = lambda _: {"data": {"result": [{"value": [0, "0"]}]}}
    assert _query_service_metric(fake, "rps_in", "orders-service") == 0.0


def test_policy_rejects_invalid_constraints():
    with pytest.raises(ValueError):
        PolicyConfig(r_min=5, r_max=2)
    with pytest.raises(ValueError):
        PolicyConfig(smoothing_alpha=2.0)
    with pytest.raises(ValueError):
        PolicyConfig(lambda_slo=-1.0)


def test_api_graph_payload(monkeypatch):
    from main import create_app
    from app.controller.tgat_controller import tgat_service

    async def predict(gw):
        return {"window": gw.window, "nodes": len(gw.nodes)}

    monkeypatch.setattr(tgat_service, "predict", predict)
    with TestClient(create_app()) as client:
        response = client.post("/api/predict", json=graph().model_dump())
        assert response.status_code == 200 and response.json()["nodes"] == 2


def test_service_does_not_commit_failed_actuation():
    from app.service.tgat_service import TGATService
    from app.core.interfaces import Prediction

    class FakeModel:
        def build_graph_from_payload(self, gw):
            return None, None, None

        def predict_targets(self, *args):
            return [
                Prediction(
                    id="orders-service",
                    rps=20,
                    replicas=4,
                    cpu_milli=800,
                    mem_mib=512,
                    p95_ms=150,
                    p99_ms=200,
                    slo_risk=0.1,
                )
            ]

        def predictions_to_actions(self, p):
            return [Action(id="orders-service", replicas=4, cpu="800m", mem="512Mi")]

        def get_last_hidden_edges(self):
            return []

        def ablation_state(self):
            return {}

    class FailedActuator:
        def apply(self, actions):
            return {
                "dry_run": False,
                "patched": [],
                "details": [{"errors": ["failure"]}],
            }

    store = MemoryStateStore()
    p = SafetyPolicy(
        PolicyConfig(dry_run=False), state_store=store, actuator=FailedActuator()
    )
    service = TGATService(model=FakeModel(), policy_factory=lambda: p)
    result = asyncio.run(service.apply(graph()))
    assert result["applied"] == []
    assert not store.load().get("last_actions", {})
    assert "last_window" not in store.load()


def test_ablation_flags_are_effective():
    from app.service.tgat_service import TGATService

    service = TGATService(model=model())
    assert service.ablate(safety_enabled=False)["safety_enabled"] is False
    assert service.ablate(safety_enabled=True)["safety_enabled"] is True


def test_unsupported_attention_beta_is_explicit():
    from main import create_app
    from config.settings import CONFIG

    before = copy.deepcopy(CONFIG)
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/sensitivity/config", json={"beta": 0.2, "history_minutes": 15}
        )
    assert response.status_code == 400
    assert CONFIG == before
