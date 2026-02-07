"""
Benchmarking harness: run experiments, collect metrics, display results.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from locality.node import TransferStats, _fmt_bytes


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    strategy: str
    total_time: float
    transfer_time: float
    compute_time: float
    bytes_transferred: int
    result_value: Any = None

    def __repr__(self):
        return (
            f"  {self.strategy:20s} | "
            f"total={self.total_time*1000:8.2f}ms | "
            f"transfer={self.transfer_time*1000:8.2f}ms | "
            f"compute={self.compute_time*1000:8.2f}ms | "
            f"bytes={_fmt_bytes(self.bytes_transferred)}"
        )


@dataclass
class ExperimentResult:
    """Collected results from running an experiment with multiple strategies."""
    name: str
    description: str
    results: list[BenchmarkResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def print_report(self):
        print(f"\n{'='*80}")
        print(f"  EXPERIMENT: {self.name}")
        print(f"  {self.description}")
        print(f"{'='*80}")

        if self.metadata:
            for k, v in self.metadata.items():
                if isinstance(v, int) and v > 1024:
                    print(f"  {k}: {_fmt_bytes(v)}")
                else:
                    print(f"  {k}: {v}")
            print(f"{'─'*80}")

        print(f"\n  {'Strategy':20s} | {'Total':>12s} | {'Transfer':>12s} | "
              f"{'Compute':>12s} | {'Bytes Moved':>12s}")
        print(f"  {'─'*20}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}")

        for r in self.results:
            print(r)

        # Compare strategies
        if len(self.results) >= 2:
            print(f"\n  Comparison:")
            base = self.results[0]
            for other in self.results[1:]:
                time_ratio = base.total_time / max(other.total_time, 1e-9)
                bytes_ratio = base.bytes_transferred / max(other.bytes_transferred, 1)
                print(f"    {other.strategy} vs {base.strategy}:")
                print(f"      Time:  {time_ratio:.2f}x {'faster' if time_ratio > 1 else 'slower'}")
                print(f"      Bytes: {bytes_ratio:.1f}x {'less' if bytes_ratio > 1 else 'more'} data moved")

        print()


def run_timed(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Run a function and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_comparison(name: str, description: str,
                         strategies: dict[str, Callable],
                         runs: int = 1) -> ExperimentResult:
    """
    Run multiple strategies and compare them.

    Args:
        name: Experiment name
        description: What we're testing
        strategies: dict of strategy_name -> callable that returns BenchmarkResult
        runs: number of runs per strategy (for averaging)
    """
    experiment = ExperimentResult(name=name, description=description)

    for strategy_name, strategy_fn in strategies.items():
        if runs == 1:
            result = strategy_fn()
            experiment.add(result)
        else:
            results = [strategy_fn() for _ in range(runs)]
            # Average the numeric fields
            avg = BenchmarkResult(
                name=name,
                strategy=strategy_name,
                total_time=statistics.mean(r.total_time for r in results),
                transfer_time=statistics.mean(r.transfer_time for r in results),
                compute_time=statistics.mean(r.compute_time for r in results),
                bytes_transferred=int(statistics.mean(r.bytes_transferred for r in results)),
                result_value=results[-1].result_value,
            )
            experiment.add(avg)

    return experiment
