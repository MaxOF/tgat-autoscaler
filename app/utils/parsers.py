from datetime import datetime, timezone
import math
import re


def parse_iso8601(s: str) -> datetime:
    value = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_cpu_milli(s: str) -> float:
    value = s.strip()
    factors = {"n": 1e-6, "u": 1e-3, "m": 1.0}
    number = (
        float(value[:-1]) * factors[value[-1]]
        if value[-1:] in factors
        else float(value) * 1000
    )
    if not math.isfinite(number) or number < 0:
        raise ValueError("CPU quantity must be finite and nonnegative")
    return number


def format_cpu_milli(v: float) -> str:
    return f"{max(0,int(round(v)))}m"


def parse_mem_mib(s: str) -> float:
    match = re.fullmatch(
        r"([+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)([KMGTPE]i|[kKMGTPE]|m)?", s.strip()
    )
    if not match:
        raise ValueError(f"Invalid memory quantity: {s}")
    value, suffix = match.groups()
    factors = {
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "Ti": 2**40,
        "Pi": 2**50,
        "Ei": 2**60,
        "k": 1e3,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
        "E": 1e18,
        "m": 1e-3,
    }
    result = float(value) * factors.get(suffix, 1) / (2**20)
    if not math.isfinite(result):
        raise ValueError("Memory quantity must be finite")
    return result


def format_mem_gi_from_mib(mib: float) -> str:
    # Preserve MiB exactly; rounding to hundredths of Gi can cross safety limits.
    return f"{max(0,mib):.6f}".rstrip("0").rstrip(".") + "Mi"
