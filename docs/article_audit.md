# Deployment Evaluation and Manuscript Correspondence

TGAT-Autoscaler combines temporal interaction modeling, approximation of missing
dependencies, multi-target forecasting, and deterministic resource control.

The project author reports testing the system on a real application comprising
**50 microservices**. This establishes the stated deployment scope. Quantitative
claims about SLO compliance, latency, recovery, resource consumption, and
controller overhead must be tied to measurements from those deployment runs.

## Real-system evaluation

A deployment evaluation separates incoming workload and observation conditions
from measured application behavior. Service and interaction telemetry provide
the forecasting inputs; Kubernetes actuation reports establish which resource
updates completed.

For a 50-service deployment, the experiment record should identify:

- The service inventory, dependency structure, and deployment/container mappings.
- Cluster resources, initial allocations, and service-specific resource bounds.
- Workload regimes, offered load, and the evaluation duration.
- Collection, aggregation, forecasting, and control intervals.
- SLO thresholds, controller configuration, and checkpoint provenance.
- The mechanism used to hide dependency observations.
- Baseline configurations, repeated runs, and evaluation partitions.

These deployment-specific values should be recorded from the actual experiment
configuration. Repository defaults and example service settings should not be
presented as measured infrastructure properties.

## Measurements and interpretation

| Measurement | Acquisition and interpretation |
|---|---|
| SLO violation rate | Fraction of measured intervals violating specified latency/error thresholds |
| p95 and p99 latency | Application request-duration histograms |
| CPU and memory usage | Container runtime telemetry |
| Resource allocation | Per-container requests/limits and desired replica counts |
| Scaling changes | Confirmed resource or replica changes |
| Recovery time | Burst onset to sustained return to the admissible region |
| Forecast MAE/RMSE | Predictions compared with exact future observations |
| SLO-risk quality | Confusion counts, precision/recall/F1, AUROC when defined, and Brier score |
| Hidden-edge reconstruction | Reconstructed dependencies compared with retained reference interactions |
| Controller overhead | Collection, graph construction, inference, decision, and actuation timing |

Measured resource usage and configured allocations answer different questions.
Resource cost also requires an explicit weighting rule or provider pricing
model. A normalized resource index is not automatically a monetary cost.

Risk metrics should include class counts: a low Brier score alone can conceal
poor detection of rare violations. Reconstruction performance should be measured
separately from downstream autoscaling outcomes.

A historical telemetry replay evaluates forecasts and proposed actions against
recorded observations. Closed-loop SLO improvement and recovery require telemetry
from an application responding to applied resource decisions.

## Relationship to the manuscript

The supplied manuscript describes a 12-service application-level evaluation.
The project author's reported 50-service testing extends that scope. Results for
the two deployment sizes should retain their own configuration and measurement
provenance.

The implementation supports the manuscript's separation of forecasting and
deterministic actuation, seven prediction targets, event-age encoding, bounded
dependency reconstruction, and operational constraints.

There are specific implementation differences:

1. The manuscript's attention equations use query/key dot products and an
   edge-intensity bias. The code uses GATv2 with temporal edge attributes.
   The unsupported beta parameter is rejected explicitly.
2. Candidate similarity uses current RPS and p95 values. Historical correlations
   of node-level series described by the manuscript are not yet implemented.
3. Hidden-edge supervision uses interactions observed within the training
   partition. Pairs never observed during training are treated as negative,
   which is an assumption requiring evaluation under persistent missingness.
4. Limits bound candidate scoring and accepted edges, but candidate generation
   retains quadratic worst-case pair enumeration.
5. The SLO action objective is a capacity proxy rather than a validated causal
   action-response model. Its operational impact requires deployment measurement.

The attention operator and container resource semantics are documented in
[PyTorch Geometric GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)
and [Kubernetes resource management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

## Implementation verification

Repository verification covers causal forecasting targets, training-only
normalization, strict checkpoints, bounded event histories, empty observed
graphs, resource units, control-time cooldown, directional hysteresis,
idempotent windows, state preservation, failed actuation, and API switches.

The regression suite currently contains 31 passing tests on the dependencies
pinned in requirements.txt. Component checks establish software behavior;
deployment measurements establish application-level performance.
