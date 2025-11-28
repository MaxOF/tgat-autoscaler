from prometheus_client import Counter, Histogram, CollectorRegistry
import requests
import datetime
from typing import Dict, Any
import math

class PromClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        _PROM_AVAILABLE = True
        _REGISTRY = CollectorRegistry()
        REQ_LAT = Histogram('tgat_request_latency_seconds', 'Request latency', registry=_REGISTRY)
        PRED_TIME = Histogram('tgat_predict_seconds', 'Predict time', registry=_REGISTRY)
        TRAIN_TIME = Histogram('tgat_train_seconds', 'Train time', registry=_REGISTRY)
        APPLY_OK = Counter('tgat_apply_total', 'Apply calls', registry=_REGISTRY)
        PRED_COUNT = Counter('tgat_predict_total', 'Predict calls', registry=_REGISTRY)

    def query(self, q: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/query"
        r = requests.get(url, params={"query": q}, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Prometheus query failed: {r.status_code} {r.text}")
        return r.json()

    def query_range(self, q: str, start: datetime, end: datetime, step_sec: int) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/query_range"
        r = requests.get(url, params={
            "query": q,
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "step": step_sec,
        }, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"Prometheus range failed: {r.status_code} {r.text}")
        return r.json()

    
    candidate_queries = {
        "cpu_mcores": [
            'sum(rate(container_cpu_usage_seconds_total{container_label_io_kubernetes_pod_name=~"$SVC.*"}[2m]))*1000',
            'sum by($LBL)(rate(container_cpu_usage_seconds_total{container_label_io_kubernetes_pod_name!=""}[2m]))*1000'
        ],
        "mem_mib": [
            'sum(container_memory_working_set_bytes{container_label_io_kubernetes_pod_name=~"$SVC.*"})/(1024*1024)',
            'sum by($LBL)(container_memory_working_set_bytes{container_label_io_kubernetes_pod_name!=""})/(1024*1024)'
        ],
        "rps_in": [
            'sum(rate(requests_total_by_service{instance=~"$SVC.*"}[2m]))',
            'sum(rate(requests_duration_in_seconds_by_service_count{instance=~"$SVC.*"}[2m]))'
        ],
        "rps_out": [
            'sum(rate(http_client_requests_seconds_count{source_service=~"$SVC"}[2m]))',
            'sum(rate(http_client_requests_total{service=~"$SVC"}[2m]))'
        ],
        "p95_ms": [
            'histogram_quantile(0.95, sum by (le) (rate(requests_duration_in_seconds_by_service_bucket{instance=~"$SVC.*"}[2m])) )*1000',
        ],
        "error_rate": [
            'sum(rate(requests_total_by_service{instance=~"$SVC.*",code=~"5.*"}[2m])) / sum(rate(requests_total_by_service{instance=~"$SVC.*"}[2m]))'
        ],
        "edge_rps": [
            'sum(rate(requests_duration_in_seconds_by_service_count{source_service=~"$SRC",destination_service="$DST"}[2m]))',
            'sum(rate(requests_total_by_service{source=~"$SRC",destination="$DST"}[2m]))'
        ],
        "edge_p95_ms": [
            'histogram_quantile(0.95, sum by (le) (rate(http_client_request_duration_seconds_bucket{source_service="$SRC",destination_service="$DST"}[2m]))) * 1000'
        ],
        "edge_errors": [
            'sum(rate(http_client_requests_seconds_count{source_service="$SRC",destination_service="$DST",status=~"5.."}[2m]))'
        ]
    }
    
    serivce_label_keys = ["service", "app", "job", "kubernetes_name"]
