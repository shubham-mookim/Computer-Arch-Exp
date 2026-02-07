"""
Experiment 1: The Simplest Case - Single Function, Two Locations

Setup:
  - Node A has a large dataset
  - Node B (our compute node) wants to run a filter on it
  - Compare: fetch data vs migrate code

This is the "hello world" of locality-aware compute.
We answer: at what data size does migration start winning?
"""

import random
import cloudpickle
from locality.node import NodeNetwork, _fmt_bytes
from locality.benchmark import BenchmarkResult, ExperimentResult


def filter_above_threshold(data, threshold=0.5):
    """Simple filter: keep values above threshold."""
    return [x for x in data if x > threshold]


def complex_filter(data, threshold=0.5):
    """More compute-intensive filter with aggregation."""
    filtered = [x for x in data if x > threshold]
    if not filtered:
        return {"count": 0, "mean": 0, "max": 0, "min": 0}
    return {
        "count": len(filtered),
        "mean": sum(filtered) / len(filtered),
        "max": max(filtered),
        "min": min(filtered),
    }


def run_single_size(data_size: int, bandwidth: float = 100_000_000,
                    latency: float = 0.001) -> ExperimentResult:
    """
    Run the experiment for a single data size.
    Compare fetch-data vs migrate-code strategies.
    """
    # Build the network
    net = NodeNetwork(default_bandwidth=bandwidth, default_latency=latency)
    data_node = net.add_node("data_node")
    compute_node = net.add_node("compute_node")

    # Generate and store data on the data node
    data = [random.random() for _ in range(data_size)]
    stored_bytes = data_node.store("dataset", data)

    code_bytes = len(cloudpickle.dumps(filter_above_threshold))

    experiment = ExperimentResult(
        name=f"Basic Migration (n={data_size:,})",
        description=f"Filter {data_size:,} floats. Data={_fmt_bytes(stored_bytes)}, Code={_fmt_bytes(code_bytes)}",
        metadata={
            "data_elements": data_size,
            "data_bytes": stored_bytes,
            "code_bytes": code_bytes,
            "bandwidth": bandwidth,
            "latency": latency,
        }
    )

    # Strategy 1: Fetch data to compute node, process locally
    net.clear_logs()
    data_copy, fetch_stats = net.fetch_data("data_node", "dataset", "compute_node")
    result_fetch, compute_time_fetch = compute_node.execute_local(
        filter_above_threshold, data_copy
    )
    result_bytes_fetch = len(cloudpickle.dumps(result_fetch))

    experiment.add(BenchmarkResult(
        name="fetch",
        strategy="Fetch Data",
        total_time=fetch_stats.transfer_time + compute_time_fetch,
        transfer_time=fetch_stats.transfer_time,
        compute_time=compute_time_fetch,
        bytes_transferred=stored_bytes,
        result_value=len(result_fetch),
    ))

    # Strategy 2: Migrate code to data node, process there, return result
    net.clear_logs()
    result_migrate, migrate_stats = net.migrate_and_execute(
        filter_above_threshold, "data_node", "dataset", "compute_node"
    )

    experiment.add(BenchmarkResult(
        name="migrate",
        strategy="Migrate Code",
        total_time=migrate_stats.total_time,
        transfer_time=migrate_stats.transfer_time,
        compute_time=migrate_stats.compute_time,
        bytes_transferred=migrate_stats.bytes_sent + migrate_stats.bytes_received,
        result_value=len(result_migrate),
    ))

    # Verify correctness
    assert len(result_fetch) == len(result_migrate), \
        f"Results differ! fetch={len(result_fetch)}, migrate={len(result_migrate)}"

    return experiment


def run_crossover_search(bandwidth: float = 100_000_000,
                         latency: float = 0.001) -> ExperimentResult:
    """
    Find the crossover point: at what data size does migration beat fetching?
    """
    sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 500_000]

    experiment = ExperimentResult(
        name="Crossover Search",
        description="Finding where migration beats fetching",
        metadata={"bandwidth": bandwidth, "latency": latency}
    )

    print(f"\n{'─'*70}")
    print(f"  Crossover Search: varying data size")
    print(f"  Bandwidth={_fmt_bytes(int(bandwidth))}/s, Latency={latency*1000:.1f}ms")
    print(f"{'─'*70}")
    print(f"  {'Size':>10s} | {'Fetch (ms)':>12s} | {'Migrate (ms)':>12s} | {'Winner':>10s} | {'Bytes Ratio':>12s}")
    print(f"  {'─'*10}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*12}")

    crossover_found = None

    for size in sizes:
        net = NodeNetwork(default_bandwidth=bandwidth, default_latency=latency)
        data_node = net.add_node("data_node")
        compute_node = net.add_node("compute_node")

        data = [random.random() for _ in range(size)]
        stored = data_node.store("dataset", data)

        # Fetch strategy
        net.clear_logs()
        data_copy, fetch_stats = net.fetch_data("data_node", "dataset", "compute_node")
        _, ct = compute_node.execute_local(filter_above_threshold, data_copy)
        fetch_time = fetch_stats.transfer_time + ct
        fetch_bytes = stored

        # Migrate strategy
        net.clear_logs()
        _, migrate_stats = net.migrate_and_execute(
            filter_above_threshold, "data_node", "dataset", "compute_node"
        )
        migrate_time = migrate_stats.total_time
        migrate_bytes = migrate_stats.bytes_sent + migrate_stats.bytes_received

        winner = "MIGRATE" if migrate_time < fetch_time else "FETCH"
        ratio = fetch_bytes / max(migrate_bytes, 1)

        if winner == "MIGRATE" and crossover_found is None:
            crossover_found = size

        print(f"  {size:>10,d} | {fetch_time*1000:>12.2f} | {migrate_time*1000:>12.2f} | "
              f"{winner:>10s} | {ratio:>10.1f}x")

    if crossover_found:
        print(f"\n  >> Crossover point: migration wins at ~{crossover_found:,} elements")
    else:
        print(f"\n  >> Migration wins at all tested sizes (code is tiny compared to data)")

    return experiment


def run_bandwidth_sensitivity() -> ExperimentResult:
    """
    How does network bandwidth affect the migration decision?
    Test with a fixed data size but varying bandwidth.
    """
    bandwidths = [
        (1_000_000, "1 MB/s (slow)"),
        (10_000_000, "10 MB/s (wifi)"),
        (100_000_000, "100 MB/s (ethernet)"),
        (1_000_000_000, "1 GB/s (fast)"),
        (10_000_000_000, "10 GB/s (datacenter)"),
    ]

    data_size = 100_000  # 100k floats
    experiment = ExperimentResult(
        name="Bandwidth Sensitivity",
        description=f"Same data ({data_size:,} elements), different network speeds",
    )

    print(f"\n{'─'*70}")
    print(f"  Bandwidth Sensitivity: {data_size:,} elements")
    print(f"{'─'*70}")
    print(f"  {'Bandwidth':>20s} | {'Fetch (ms)':>12s} | {'Migrate (ms)':>12s} | {'Speedup':>10s}")
    print(f"  {'─'*20}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*10}")

    for bw, label in bandwidths:
        net = NodeNetwork(default_bandwidth=bw, default_latency=0.001)
        data_node = net.add_node("data_node")
        compute_node = net.add_node("compute_node")

        data = [random.random() for _ in range(data_size)]
        data_node.store("dataset", data)

        # Fetch
        net.clear_logs()
        data_copy, fs = net.fetch_data("data_node", "dataset", "compute_node")
        _, ct = compute_node.execute_local(filter_above_threshold, data_copy)
        fetch_time = fs.transfer_time + ct

        # Migrate
        net.clear_logs()
        _, ms = net.migrate_and_execute(
            filter_above_threshold, "data_node", "dataset", "compute_node"
        )
        migrate_time = ms.total_time

        speedup = fetch_time / max(migrate_time, 1e-9)
        print(f"  {label:>20s} | {fetch_time*1000:>12.2f} | {migrate_time*1000:>12.2f} | {speedup:>9.2f}x")

    return experiment


def run():
    """Run all Phase 1 experiments."""
    print("\n" + "="*80)
    print("  PHASE 1: Basic Migration - Single Function, Two Locations")
    print("="*80)

    # Experiment 1a: single size comparison
    exp1 = run_single_size(100_000)
    exp1.print_report()

    # Experiment 1b: find the crossover point
    run_crossover_search()

    # Experiment 1c: bandwidth sensitivity
    run_bandwidth_sensitivity()

    return exp1


if __name__ == "__main__":
    run()
