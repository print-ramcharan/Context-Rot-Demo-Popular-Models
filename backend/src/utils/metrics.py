import statistics
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timings = defaultdict(list)

    def increment(self, name: str, value: int = 1):
        self.counters[name] += value

    def observe(self, name: str, value: float):
        self.timings[name].append(float(value))

    def _percentile(self, values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = (len(values) - 1) * pct
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[int(k)]
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1

    def snapshot(self) -> dict:
        timing_stats = {}
        for name, values in self.timings.items():
            if not values:
                continue
            timing_stats[name] = {
                "count": len(values),
                "avg_ms": statistics.mean(values),
                "p50_ms": self._percentile(values, 0.5),
                "p95_ms": self._percentile(values, 0.95),
                "p99_ms": self._percentile(values, 0.99),
                "max_ms": max(values)
            }
        return {
            "counters": dict(self.counters),
            "timings": timing_stats
        }
