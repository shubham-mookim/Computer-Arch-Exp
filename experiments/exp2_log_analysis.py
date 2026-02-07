"""
Experiment 2: Distributed Log Analysis

Scenario:
  - 10 servers each have log files
  - We want to count errors across all logs
  - Compare: pull all logs centrally vs send counting function to each server

This is the canonical "embarrassingly parallel" locality-aware problem.
"""

import random
import string
import cloudpickle
from locality.node import NodeNetwork, _fmt_bytes
from locality.migration import set_runtime, DataRef
from locality.parallel import PartitionedData, merge_sum, merge_dicts
from locality.benchmark import BenchmarkResult, ExperimentResult


LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
LOG_WEIGHTS = [0.40, 0.35, 0.15, 0.08, 0.02]


def generate_log_lines(n_lines: int) -> list[str]:
    """Generate realistic-ish log lines."""
    lines = []
    for i in range(n_lines):
        level = random.choices(LOG_LEVELS, LOG_WEIGHTS)[0]
        msg = ''.join(random.choices(string.ascii_lowercase + ' ', k=random.randint(30, 120)))
        lines.append(f"2025-01-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d} [{level}] {msg}")
    return lines


def count_errors(log_lines):
    """Count lines containing ERROR or FATAL."""
    return sum(1 for line in log_lines if 'ERROR' in line or 'FATAL' in line)


def count_by_level(log_lines):
    """Count occurrences of each log level."""
    counts = {}
    for line in log_lines:
        for level in LOG_LEVELS:
            if f'[{level}]' in line:
                counts[level] = counts.get(level, 0) + 1
                break
    return counts


def grep_pattern(log_lines, pattern="ERROR"):
    """Filter lines matching a pattern (like grep)."""
    return [line for line in log_lines if pattern in line]


def run_distributed_log_analysis(n_servers: int = 10,
                                  lines_per_server: int = 50_000,
                                  bandwidth: float = 100_000_000,
                                  latency: float = 0.005) -> ExperimentResult:
    """
    Compare centralized vs distributed log analysis.
    """
    net = NodeNetwork(default_bandwidth=bandwidth, default_latency=latency)
    compute_node = net.add_node("analyzer")

    # Create servers with log data
    total_data_bytes = 0
    for i in range(n_servers):
        server = net.add_node(f"server_{i}")
        logs = generate_log_lines(lines_per_server)
        size = server.store("logs", logs)
        total_data_bytes += size

    code_size = len(cloudpickle.dumps(count_errors))

    experiment = ExperimentResult(
        name=f"Distributed Log Analysis ({n_servers} servers)",
        description=(
            f"{n_servers} servers x {lines_per_server:,} log lines each. "
            f"Total data: {_fmt_bytes(total_data_bytes)}, Code: {_fmt_bytes(code_size)}"
        ),
        metadata={
            "n_servers": n_servers,
            "lines_per_server": lines_per_server,
            "total_data_bytes": total_data_bytes,
            "code_bytes": code_size,
        }
    )

    # Strategy 1: Centralized - fetch all logs, process locally
    net.clear_logs()
    total_fetch_time = 0
    total_fetch_bytes = 0
    all_logs = []

    for i in range(n_servers):
        data, stats = net.fetch_data(f"server_{i}", "logs", "analyzer")
        all_logs.extend(data)
        total_fetch_time += stats.transfer_time
        total_fetch_bytes += stats.bytes_sent

    result_central, compute_time = compute_node.execute_local(count_errors, all_logs)

    experiment.add(BenchmarkResult(
        name="centralized",
        strategy="Centralized (pull all)",
        total_time=total_fetch_time + compute_time,
        transfer_time=total_fetch_time,
        compute_time=compute_time,
        bytes_transferred=total_fetch_bytes,
        result_value=result_central,
    ))

    # Strategy 2: Distributed - send counting function to each server
    net.clear_logs()
    total_migrate_time = 0
    total_migrate_bytes = 0
    total_errors = 0

    for i in range(n_servers):
        result, stats = net.migrate_and_execute(
            count_errors, f"server_{i}", "logs", "analyzer"
        )
        total_errors += result
        total_migrate_time += stats.total_time  # includes transfer + compute
        total_migrate_bytes += stats.bytes_sent + stats.bytes_received

    # In reality, migrations run in parallel. Simulate parallel time:
    # parallel_time = max of individual times (not sum)
    # But we also track sequential for comparison
    parallel_migrate_time = max(
        s.total_time for s in net.transfer_log
    )

    experiment.add(BenchmarkResult(
        name="distributed_seq",
        strategy="Distributed (sequential)",
        total_time=total_migrate_time,
        transfer_time=sum(s.transfer_time for s in net.transfer_log),
        compute_time=sum(s.compute_time for s in net.transfer_log),
        bytes_transferred=total_migrate_bytes,
        result_value=total_errors,
    ))

    experiment.add(BenchmarkResult(
        name="distributed_par",
        strategy="Distributed (parallel)",
        total_time=parallel_migrate_time,
        transfer_time=max(s.transfer_time for s in net.transfer_log),
        compute_time=max(s.compute_time for s in net.transfer_log),
        bytes_transferred=total_migrate_bytes,
        result_value=total_errors,
    ))

    # Verify correctness
    assert result_central == total_errors, \
        f"Results differ! central={result_central}, distributed={total_errors}"

    return experiment


def run_scaling_analysis():
    """How does the advantage scale with number of servers and data size?"""
    print(f"\n{'─'*70}")
    print(f"  Scaling Analysis: varying servers and data size")
    print(f"{'─'*70}")
    print(f"  {'Servers':>8s} | {'Lines/srv':>10s} | {'Fetch (ms)':>12s} | {'Migrate (ms)':>12s} | {'Speedup':>10s} | {'Bytes Saved':>12s}")
    print(f"  {'─'*8}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*12}")

    configs = [
        (3, 10_000),
        (5, 10_000),
        (10, 10_000),
        (10, 50_000),
        (10, 100_000),
        (20, 50_000),
    ]

    for n_servers, lines in configs:
        net = NodeNetwork(default_bandwidth=100_000_000, default_latency=0.005)
        compute = net.add_node("analyzer")

        for i in range(n_servers):
            srv = net.add_node(f"s{i}")
            srv.store("logs", generate_log_lines(lines))

        # Fetch all
        net.clear_logs()
        fetch_time = 0
        fetch_bytes = 0
        all_data = []
        for i in range(n_servers):
            d, s = net.fetch_data(f"s{i}", "logs", "analyzer")
            all_data.extend(d)
            fetch_time += s.transfer_time
            fetch_bytes += s.bytes_sent
        _, ct = compute.execute_local(count_errors, all_data)
        fetch_time += ct

        # Migrate (parallel = max time)
        net.clear_logs()
        migrate_bytes = 0
        for i in range(n_servers):
            _, s = net.migrate_and_execute(count_errors, f"s{i}", "logs", "analyzer")
            migrate_bytes += s.bytes_sent + s.bytes_received
        migrate_time = max(s.total_time for s in net.transfer_log)

        speedup = fetch_time / max(migrate_time, 1e-9)
        bytes_saved_pct = (1 - migrate_bytes / max(fetch_bytes, 1)) * 100

        print(f"  {n_servers:>8d} | {lines:>10,d} | {fetch_time*1000:>12.2f} | "
              f"{migrate_time*1000:>12.2f} | {speedup:>9.1f}x | {bytes_saved_pct:>10.1f}%")


def run():
    """Run all Phase 2 experiments."""
    print("\n" + "="*80)
    print("  PHASE 2: Distributed Log Analysis")
    print("="*80)

    exp = run_distributed_log_analysis()
    exp.print_report()

    run_scaling_analysis()

    return exp


if __name__ == "__main__":
    run()
