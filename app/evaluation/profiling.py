# app/evaluation/profiling.py

import os
import time
import psutil


class RuntimeProfiler:

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def measure(self, fn, *args, **kwargs):

        rss_before = self.process.memory_info().rss

        cpu_before = self.process.cpu_times()

        t0 = time.perf_counter()

        result = fn(*args, **kwargs)

        elapsed = time.perf_counter() - t0

        cpu_after = self.process.cpu_times()

        rss_after = self.process.memory_info().rss

        cpu_sec = (
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
        )

        return result, {
            "runtime_ms": elapsed * 1000.0,
            "cpu_time_ms": cpu_sec * 1000.0,
            "rss_before_mb": rss_before / 1024**2,
            "rss_after_mb": rss_after / 1024**2,
            "rss_delta_mb": (rss_after - rss_before) / 1024**2,
        }
