# TGAT-Autoscaler

TGAT-Autoscaler combines temporal graph forecasting with deterministic horizontal
and vertical resource control for Kubernetes microservices. It represents observed
service interactions as time-stamped edges, estimates missing dependencies, and
predicts future workload, resource demand, tail latency, and SLO risk.

Project testing covered a real system containing **50 microservices**, as reported
by the project author. This document describes the implemented architecture and
deployment workflow. Quantitative evaluation results belong to the corresponding
deployment measurements.

## Control pipeline

1. Collect service and interaction telemetry from Prometheus.
2. Build a temporal graph from events within the configured observation window.
3. Encode event ages and compute service representations.
4. Score candidate missing dependencies and construct an extended graph.
5. Forecast request rate, replicas, CPU, memory, p95, p99, and SLO risk.
6. Evaluate resource candidates and enforce operational constraints.
7. Apply Kubernetes updates and record their outcomes.

The default decision interval and forecasting horizon are five minutes. The
observation window defaults to 60 minutes. These are repository defaults and
should be checked against the deployment configuration.

## Installation and startup

Python 3.11–3.13 is supported. Repository checks have been run on Windows with
Python 3.13 and the dependencies pinned in requirements.txt.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe main.py
~~~

The application entry point is main.py. The legacy tgat_autoscaler_service.py
entry point delegates to the same application.

## Configure the deployment

Set the Prometheus URL, service inventory, dependency priors, resource bounds,
SLO thresholds, and control parameters in config/settings.py. Populate the
service inventory with the deployments in the target system.

The checked-in inventory contains three example services. It is a configuration
example rather than the complete inventory of the reported 50-service system.

A compatible trained checkpoint is required for normal inference. The default
model path is artifacts/tgat_model.pt. Schema-v2 checkpoints store the network
configuration, feature and target orders, training normalization, target scales,
and validation metadata. Earlier checkpoint formats require retraining.

Kubernetes credentials are loaded from the in-cluster service account or local
kubeconfig. Deployment settings include:

- TGAT_NAMESPACE: deployment namespace; defaults to default.
- TGAT_CONTAINER: workload container name; required to resolve a multi-container deployment.
- TGAT_STATE_PATH: policy state file.
- TGAT_FIELD_MANAGER: manager identifier used for resource patches.

The default policy uses dry_run. Set the deployment's safety.dry_run configuration
to false when applying actual resource changes.

An external scheduler must invoke POST /api/apply at the configured decision
interval. The application currently exposes the control operation without
starting an automatic background schedule.

## Telemetry and training

Node telemetry includes CPU usage, memory usage, incoming and outgoing request
rates, p95 and p99 latency, error rate, and replica count. Allocated CPU and memory
requests are collected separately from observed resource usage.

Observed interactions retain their event timestamps, operational attributes, and
confidence. Architectural priors assist dependency reconstruction and are kept
distinct from measured interactions.

Training accepts exported node and edge telemetry through POST /api/train_csv.
Node records identify the service and observation time. Edge records identify
source, destination, and event time. Resource target columns must have documented
deployment-specific semantics; an observed allocation is a reference label and
does not automatically establish an optimal allocation.

The CSV adapter aligns all seven targets to the exact forecasting horizon.
Training uses chronological partitions, excludes targets crossing the partition
boundary, and fits normalization exclusively on training observations. Graphs
are processed in batches, and the checkpoint is selected using validation loss.

Historical telemetry can also be replayed to evaluate forecasts and proposed
actions. A replay evaluates recorded observations; deployment-level effects of
actions must be measured in the running application.

## API

| Endpoint | Purpose |
|---|---|
| GET /api/health | Service health |
| GET /api/config | Effective configuration |
| POST /api/train | Training with configured telemetry paths |
| POST /api/train_csv | Training from supplied CSV paths |
| POST /api/predict | Forecasting; optional GraphWindow body |
| POST /api/apply | Forecasting, policy evaluation, and actuation |
| POST /api/ablate | Configure experimental switches |
| POST /api/ablate/reset | Restore experiment switches |
| POST /api/sensitivity/config | Configure supported sensitivity parameters |
| GET /api/hidden-edges | Latest reconstructed dependencies |

Without a graph body, prediction and control operations collect live telemetry.
Observed event history is retained across calls and requires warm-up after a
restart.

## Operational behavior

Policy decisions enforce service bounds, replica and resource step limits,
cooldown, directional hysteresis, and smoothing. The control clock follows graph
timestamps, and repeated evaluation of the same window is idempotent.

State files are replaced atomically. Failed actuation is distinguished from
successful updates; a wholly failed operation does not advance the policy window.
Partial Kubernetes updates remain visible in the actuation report.

Operations sharing model state are serialized and blocking work executes outside
the API event loop. Use one API process with the file-backed state repository.
Multiple processes require a shared transactional state implementation.

The current attention implementation uses GATv2. The beta attention-bias parameter
from the manuscript is explicitly rejected because this implementation does not
provide that operator.

See [architecture](docs/architecture.md) and
[deployment evaluation and manuscript correspondence](docs/article_audit.md).
