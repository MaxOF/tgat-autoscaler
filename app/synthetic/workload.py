"""Workload and telemetry model for synthetic software verification.
These functions generate simulated outputs, not measurements from Kubernetes.
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np

MSK_OFFSET = timedelta(hours=3)
RPS_PER_CORE = 30.0
RPS_PER_REPLICA = 60.0
SLO_P95_MS = 400.0
SLO_P99_MS = 600.0
SLO_ERROR_RATE = 0.01
BASE_EDGE_SHARE = {
    ("orders-service", "products-service"): 0.6,
    ("orders-service", "payments-service"): 0.4,
}


class SyntheticWorkload:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def scenario_for_day(self, day_index: int) -> str:
        if day_index <= 5:
            return "weakly_varying"
        if day_index <= 11:
            return "diurnal"
        if day_index <= 17:
            return "non_stationary"
        if day_index <= 23:
            return "flash_crowd"
        return "partial_observability"

    def msk_time(self, dt_utc: datetime) -> datetime:
        return dt_utc + MSK_OFFSET

    def daily_phase(self, dt_utc: datetime) -> float:
        msk = self.msk_time(dt_utc)
        minutes = msk.hour * 60 + msk.minute
        return minutes / (24.0 * 60.0)

    def weak_multiplier(self, dt_utc: datetime) -> float:
        """
        Low-amplitude workload variation.
        """
        phase = self.daily_phase(dt_utc)
        periodic = 0.03 * math.sin(2.0 * math.pi * phase)
        return 1.0 + periodic

    def diurnal_multiplier(self, dt_utc: datetime) -> float:
        """
        Pronounced periodic daily demand.
        """
        phase = self.daily_phase(dt_utc)
        periodic = 0.45 * math.sin(2.0 * math.pi * (phase - 0.2))
        return max(0.45, 1.0 + periodic)

    def non_stationary_multiplier(self, dt_utc: datetime, day_index: int) -> float:
        """
        Diurnal load with a trend and regime shift.
        """
        base = self.diurnal_multiplier(dt_utc)
        local_day = day_index - 12
        trend = 1.0 + 0.07 * max(0, local_day)
        regime_shift = 1.35 if day_index >= 15 else 1.0
        return base * trend * regime_shift

    def is_flash_event(self, dt_utc: datetime, day_index: int) -> bool:
        """
        Several repeatable flash-crowd bursts inside days 18--23.
        """
        if not 18 <= day_index <= 23:
            return False
        msk = self.msk_time(dt_utc)
        flash_days = {18, 20, 22, 23}
        return day_index in flash_days and 18 <= msk.hour < 20

    def flash_multiplier(self, dt_utc: datetime, day_index: int) -> float:
        base = self.diurnal_multiplier(dt_utc)
        if self.is_flash_event(dt_utc, day_index):
            return base * self.rng.uniform(4.5, 7.0)
        return base

    def scenario_multiplier(
        self, dt_utc: datetime, day_index: int, scenario: str
    ) -> float:
        if scenario == "weakly_varying":
            return self.weak_multiplier(dt_utc)
        if scenario == "diurnal":
            return self.diurnal_multiplier(dt_utc)
        if scenario == "non_stationary":
            return self.non_stationary_multiplier(dt_utc, day_index)
        if scenario == "flash_crowd":
            return self.flash_multiplier(dt_utc, day_index)
        return self.diurnal_multiplier(dt_utc)

    def cpu_from_rps(self, rps: float) -> float:
        expected = rps / RPS_PER_CORE * 1000.0
        noise = self.rng.normal(0.0, max(5.0, expected * 0.08))
        return max(50.0, expected + noise)

    def memory_from_rps(
        self, rps: float, base: float = 512.0, slope: float = 0.8
    ) -> float:
        noise = self.rng.normal(0.0, 35.0)
        return max(256.0, base + slope * rps + noise)

    def utilization_pressure(self, rps: float) -> float:
        """
        Smooth overload pressure used to make latency/nonlinear error
        behavior more realistic.
        """
        reference = 60.0
        ratio = rps / reference
        return max(0.0, ratio - 1.0)

    def p95_from_rps(self, rps: float, *, service: str, flash: bool) -> float:
        service_base = {
            "orders-service": 125.0,
            "products-service": 105.0,
            "payments-service": 135.0,
        }.get(service, 120.0)
        pressure = self.utilization_pressure(rps)
        nonlinear = 55.0 * pressure * pressure
        flash_penalty = 55.0 if flash else 0.0
        noise = self.rng.normal(0.0, 8.0)
        return float(
            np.clip(
                service_base
                + 5.0 * math.log1p(max(0.0, rps))
                + nonlinear
                + flash_penalty
                + noise,
                40.0,
                1200.0,
            )
        )

    def p99_from_p95(self, p95_ms: float, *, flash: bool) -> float:
        factor = self.rng.uniform(1.25, 1.4)
        if flash:
            factor += 0.1
        noise = self.rng.normal(0.0, 10.0)
        return float(np.clip(p95_ms * factor + noise, p95_ms, 1800.0))

    def error_rate_from_rps(self, rps: float, *, flash: bool) -> float:
        pressure = self.utilization_pressure(rps)
        rate = 0.0025 + 0.006 * pressure + abs(self.rng.normal(0.0, 0.001))
        if flash:
            rate += 0.004
        return float(np.clip(rate, 0.0, 0.12))

    def targets_from_load(
        self, rps: float, p95_ms: float, p99_ms: float, error_rate: float
    ) -> Tuple[int, float, float]:
        """
        Deterministic oracle-like target used for supervised synthetic data.

        This is not the TGAT controller. It only produces labels for the
        synthetic training dataset.
        """
        risk_factor = 1.0
        if p95_ms > SLO_P95_MS:
            risk_factor += 0.15
        if p99_ms > SLO_P99_MS:
            risk_factor += 0.15
        if error_rate > SLO_ERROR_RATE:
            risk_factor += 0.1
        replicas = max(1, int(math.ceil(rps * risk_factor / RPS_PER_REPLICA)))
        cpu_m = max(200.0, rps / RPS_PER_CORE * 1000.0 * 1.15 * risk_factor)
        mem_mib = max(512.0, 512.0 + 0.85 * rps * risk_factor)
        return (replicas, cpu_m, mem_mib)

    def slo_risk_label(self, p95_ms: float, p99_ms: float, error_rate: float) -> int:
        return int(
            p95_ms > SLO_P95_MS or p99_ms > SLO_P99_MS or error_rate > SLO_ERROR_RATE
        )

    def outgoing_shares(self) -> Tuple[float, float]:
        """
        Generate noisy but normalized orders -> {products, payments} traffic
        proportions.
        """
        products_share = max(
            0.01,
            BASE_EDGE_SHARE["orders-service", "products-service"]
            + self.rng.normal(0.0, 0.03),
        )
        payments_share = max(
            0.01,
            BASE_EDGE_SHARE["orders-service", "payments-service"]
            + self.rng.normal(0.0, 0.03),
        )
        total = products_share + payments_share
        return (products_share / total, payments_share / total)

    def edge_metrics(
        self,
        src: str,
        dst: str,
        edge_share: float,
        per_service: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        src_out = float(per_service[src]["rps_out"])
        dst_p95 = float(per_service[dst]["p95_ms"])
        dst_error = float(per_service[dst]["error_rate"])
        edge_rps = max(0.0, src_out * edge_share * self.rng.uniform(0.97, 1.03))
        edge_p95 = max(1.0, dst_p95 * self.rng.uniform(0.9, 1.05))
        edge_errors = max(0.0, edge_rps * dst_error)
        return {
            "edge_weight": float(edge_share),
            "edge_rps": float(edge_rps),
            "edge_p95_ms": float(edge_p95),
            "edge_errors": float(edge_errors),
        }
