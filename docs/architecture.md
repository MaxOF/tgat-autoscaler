# Architecture

The implementation separates telemetry, forecasting, dependency reconstruction,
policy evaluation, state persistence, and Kubernetes actuation. It uses adapters,
a state repository, dependency injection, and a service facade.

~~~mermaid
flowchart LR
    P[PrometheusGraphProvider] --> S[TGATService]
    C[CSVGraphDataset] --> T[CSVTrainer]
    C --> R[Telemetry replay]
    T --> M[TGATAutoscalerModel]
    R --> M
    S --> M
    M --> N[SimpleTGAT]
    M --> H[HiddenEdgeReconstructor]
    S --> D[SafetyPolicy]
    R --> D
    D --> A[KubernetesActuator]
    D --> J[StateStore]
~~~

## Components

| Module | Responsibility |
|---|---|
| app/core/graph.py | Prometheus collection and retained event history |
| app/core/network.py | Neural architecture and trainable prediction heads |
| app/core/model.py | Tensor preparation, checkpoint loading, and forecasting |
| app/core/hidden_edges.py | Candidate filtering, scoring, and bounded reconstruction |
| app/data/csv_dataset.py | Telemetry indexing, causal history, and future targets |
| app/training/trainer.py | Temporal partitioning, batched training, and checkpoints |
| app/core/safety.py | Candidate scoring and operational constraints |
| app/core/state.py | StateStore interface and memory/file repositories |
| app/core/actuation.py | Kubernetes resource and replica updates |
| app/service/tgat_service.py | Pipeline orchestration and actuation outcome handling |
| app/controller/tgat_controller.py | HTTP interface and operation serialization |
| run_offline.py | Historical telemetry training and replay utilities |

TGATService accepts a model, graph provider, and policy factory. SafetyPolicy
accepts a state store and an actuator. These boundaries allow external systems
to be replaced in component tests and permit alternative persistence adapters.

## Temporal invariants

Timestamps include a timezone and are normalized to UTC. Observed events belong
to the interval [t-W, t). Future and expired events are excluded, and incoming
temporal neighborhoods have a configurable budget.

Forecasting targets are taken from the exact time t + horizon. Missing future
observations are not replaced with current values or the next available row.

An empty observed graph remains empty. Configured dependencies are reconstruction
priors rather than measured interactions. Empty neighborhoods use the trained
local graph path.

Feature normalization is fitted on the training partition and applied once.
Checkpoint loading checks feature order, target order, architecture, and model
weights strictly. Untrained inference requires an explicit development setting.

## Control and state

Horizontal constraints begin from observed replica counts. Resource requests
are distinct from measured CPU and memory consumption. Vertical step limits use
available allocation metadata or previously recorded allocations.

Directional hysteresis requires consecutive proposals in the same direction.
Cooldown uses observation timestamps. Repeated windows do not advance policy
state, and updating one service preserves other services' state.

The file repository writes a temporary file and replaces the destination
atomically. This prevents truncated JSON but does not provide transactions
between independent API processes.

The actuator identifies the workload container and reports horizontal and
vertical outcomes separately. An operation is considered fully successful only
when its required updates complete without errors. Subsequent telemetry must
reconcile the actual configuration after partial updates.

## Deployment scope

The project author reports testing on a real system of 50 microservices.
Deploying the repository against that system requires its actual service
inventory, namespace/container mappings, telemetry labels, and resource bounds.

Computational budgets limit sampled temporal neighbors, scored candidates, and
accepted hidden edges. Candidate generation still examines service pairs and has
quadratic worst-case complexity. Deployment overhead must therefore be measured
against actual graph sizes and event volumes.

## Model boundaries

The SLO objective uses resource sufficiency as a deterministic capacity proxy.
It is not a learned causal model of how a resource action changes latency.

Candidate actions primarily vary replicas. CPU and memory are proposed by the
forecast and projected onto resource constraints. A complete joint optimizer
for all three resources is a separate extension.

GPU execution remains supported by the code. The repository regression checks
were run on CPU.
