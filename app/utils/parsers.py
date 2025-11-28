from datetime import datetime, timezone, timedelta


def parse_iso8601(s: str) -> datetime:
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)

def now_utc_iso() -> str:
    msk_offset = timezone(timedelta(hours=3))
    return datetime.now(msk_offset).replace(microsecond=0).isoformat()

def parse_cpu_milli(s: str) -> float:
    # "800m" → 800, "2" → 2000
    s = s.strip().lower()
    if s.endswith('m'):
        return float(s[:-1])
  
    return float(s) * 1000.0

def format_cpu_milli(v: float) -> str:
    return f"{int(round(v))}m"

def parse_mem_mib(s: str) -> float:
    s = s.strip()
    ls = s.lower()
    if ls.endswith('g') or ls.endswith('gi'):
        num = float(s[:-1]) if ls.endswith('g') else float(s[:-2])
        return num * 1024.0
    if ls.endswith('mi'):
        return float(s[:-2])
    if ls.endswith('m'):
        return float(s[:-1])
    if ls.endswith('ki'):
        return float(s[:-2]) / 1024.0
    # bytes → MiB
    try:
        return float(s) / (1024.0 * 1024.0)
    except Exception:
        return 0.0

def format_mem_gi_from_mib(mib: float) -> str:
    gi = max(0.0, mib / 1024.0)
    return f"{gi:.2f}Gi"