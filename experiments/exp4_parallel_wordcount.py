"""
Experiment 4: Parallel Word Count with Auto-Migration

The classic MapReduce problem, but with automatic code migration.

Scenario:
  - Text corpus split across N servers
  - Send word_count function to each server (map phase)
  - Merge results (reduce phase)
  - Compare: centralized vs distributed vs parallel distributed

Also explores:
  - Uneven data distribution (skew)
  - Slow/straggler nodes
  - Different merge strategies
"""

import random
import string
import time
import cloudpickle
from locality.node import NodeNetwork, _fmt_bytes
from locality.migration import set_runtime
from locality.parallel import PartitionedData, auto_parallelize, merge_dicts
from locality.benchmark import BenchmarkResult, ExperimentResult


# Sample words to generate realistic-ish text
WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "data", "compute", "network", "migration", "locality", "function", "node",
    "server", "distributed", "parallel", "error", "warning", "process", "system",
]


def generate_text(n_words: int) -> list[str]:
    """Generate text as a list of words."""
    return [random.choice(WORDS) for _ in range(n_words)]


def word_count(text):
    """Count word frequencies in a list of words."""
    counts = {}
    for word in text:
        counts[word] = counts.get(word, 0) + 1
    return counts


def word_count_with_filter(text, min_length=3):
    """Count only words above a minimum length."""
    counts = {}
    for word in text:
        if len(word) >= min_length:
            counts[word] = counts.get(word, 0) + 1
    return counts


def merge_word_counts(count_dicts: list[dict]) -> dict:
    """Merge multiple word count dictionaries."""
    merged = {}
    for d in count_dicts:
        for word, count in d.items():
            merged[word] = merged.get(word, 0) + count
    return merged


def run_parallel_wordcount(n_nodes: int = 10,
                           words_per_node: int = 100_000,
                           bandwidth: float = 100_000_000,
                           latency: float = 0.005) -> ExperimentResult:
    """
    Compare centralized vs distributed word count.
    """
    net = NodeNetwork(default_bandwidth=bandwidth, default_latency=latency)
    compute = net.add_node("aggregator")

    total_data_bytes = 0
    for i in range(n_nodes):
        node = net.add_node(f"shard_{i}")
        text = generate_text(words_per_node)
        size = node.store("text", text)
        total_data_bytes += size

    code_size = len(cloudpickle.dumps(word_count))
    merge_code_size = len(cloudpickle.dumps(merge_word_counts))

    experiment = ExperimentResult(
        name=f"Parallel Word Count ({n_nodes} shards)",
        description=(
            f"{n_nodes} shards x {words_per_node:,} words each. "
            f"Total data: {_fmt_bytes(total_data_bytes)}, Code: {_fmt_bytes(code_size)}"
        ),
        metadata={
            "n_nodes": n_nodes,
            "words_per_node": words_per_node,
            "total_words": n_nodes * words_per_node,
            "total_data_bytes": total_data_bytes,
            "code_bytes": code_size,
        }
    )

    # Strategy 1: Centralized - fetch all text, count locally
    net.clear_logs()
    all_text = []
    fetch_transfer = 0
    fetch_bytes = 0
    for i in range(n_nodes):
        data, stats = net.fetch_data(f"shard_{i}", "text", "aggregator")
        all_text.extend(data)
        fetch_transfer += stats.transfer_time
        fetch_bytes += stats.bytes_sent

    result_central, compute_time = compute.execute_local(word_count, all_text)

    experiment.add(BenchmarkResult(
        name="centralized",
        strategy="Centralized",
        total_time=fetch_transfer + compute_time,
        transfer_time=fetch_transfer,
        compute_time=compute_time,
        bytes_transferred=fetch_bytes,
        result_value=sum(result_central.values()),
    ))

    # Strategy 2: Distributed sequential - migrate word_count to each shard
    net.clear_logs()
    partial_counts = []
    migrate_transfer = 0
    migrate_bytes = 0
    migrate_compute = 0

    for i in range(n_nodes):
        result, stats = net.migrate_and_execute(
            word_count, f"shard_{i}", "text", "aggregator"
        )
        partial_counts.append(result)
        migrate_transfer += stats.transfer_time
        migrate_bytes += stats.bytes_sent + stats.bytes_received
        migrate_compute += stats.compute_time

    # Merge phase
    merged_result, merge_time = compute.execute_local(merge_word_counts, partial_counts)

    experiment.add(BenchmarkResult(
        name="distributed_seq",
        strategy="Distributed (sequential)",
        total_time=migrate_transfer + migrate_compute + merge_time,
        transfer_time=migrate_transfer,
        compute_time=migrate_compute + merge_time,
        bytes_transferred=migrate_bytes,
        result_value=sum(merged_result.values()),
    ))

    # Strategy 3: Distributed parallel - all migrations happen at once
    # In parallel, transfer time = max of individual transfers
    net.clear_logs()
    par_counts = []
    par_bytes = 0
    max_par_time = 0

    for i in range(n_nodes):
        result, stats = net.migrate_and_execute(
            word_count, f"shard_{i}", "text", "aggregator"
        )
        par_counts.append(result)
        par_bytes += stats.bytes_sent + stats.bytes_received
        max_par_time = max(max_par_time, stats.total_time)

    par_merged, par_merge_time = compute.execute_local(merge_word_counts, par_counts)

    experiment.add(BenchmarkResult(
        name="distributed_par",
        strategy="Distributed (parallel)",
        total_time=max_par_time + par_merge_time,
        transfer_time=max_par_time - max(s.compute_time for s in net.transfer_log),
        compute_time=max(s.compute_time for s in net.transfer_log) + par_merge_time,
        bytes_transferred=par_bytes,
        result_value=sum(par_merged.values()),
    ))

    # Verify
    assert sum(result_central.values()) == sum(merged_result.values()), "Results differ!"

    return experiment


def run_skew_analysis():
    """
    What happens when data is unevenly distributed?
    One node has 90% of the data, others have 10%.
    """
    print(f"\n{'─'*70}")
    print(f"  Skew Analysis: uneven data distribution")
    print(f"{'─'*70}")

    total_words = 500_000
    n_nodes = 10

    configs = [
        ("Even", [total_words // n_nodes] * n_nodes),
        ("Mild skew", [total_words // 5] + [total_words * 4 // (5 * (n_nodes - 1))] * (n_nodes - 1)),
        ("Heavy skew", [total_words * 9 // 10] + [total_words // (10 * (n_nodes - 1))] * (n_nodes - 1)),
        ("One node", [total_words] + [100] * (n_nodes - 1)),
    ]

    print(f"  {'Distribution':>15s} | {'Fetch (ms)':>12s} | {'Migrate Par (ms)':>16s} | {'Speedup':>10s}")
    print(f"  {'─'*15}─┼─{'─'*12}─┼─{'─'*16}─┼─{'─'*10}")

    for label, distribution in configs:
        net = NodeNetwork(default_bandwidth=100_000_000, default_latency=0.005)
        compute = net.add_node("agg")

        for i, words in enumerate(distribution):
            node = net.add_node(f"s{i}")
            node.store("text", generate_text(max(words, 1)))

        # Fetch all
        net.clear_logs()
        all_text = []
        for i in range(n_nodes):
            d, _ = net.fetch_data(f"s{i}", "text", "agg")
            all_text.extend(d)
        _, ct = compute.execute_local(word_count, all_text)
        fetch_time = sum(s.transfer_time for s in net.transfer_log) + ct

        # Migrate parallel
        net.clear_logs()
        partials = []
        for i in range(n_nodes):
            r, _ = net.migrate_and_execute(word_count, f"s{i}", "text", "agg")
            partials.append(r)
        max_time = max(s.total_time for s in net.transfer_log)
        _, mt = compute.execute_local(merge_word_counts, partials)
        migrate_time = max_time + mt

        speedup = fetch_time / max(migrate_time, 1e-9)
        print(f"  {label:>15s} | {fetch_time*1000:>12.2f} | {migrate_time*1000:>16.2f} | {speedup:>9.1f}x")


def run_straggler_analysis():
    """
    What happens when one node is significantly slower?
    """
    print(f"\n{'─'*70}")
    print(f"  Straggler Analysis: one slow node")
    print(f"{'─'*70}")

    n_nodes = 10
    words_per_node = 50_000

    print(f"  {'Slowdown':>12s} | {'Parallel (ms)':>14s} | {'Impact':>10s}")
    print(f"  {'─'*12}─┼─{'─'*14}─┼─{'─'*10}")

    baseline = None
    for slow_factor in [1.0, 2.0, 5.0, 10.0, 20.0]:
        net = NodeNetwork(default_bandwidth=100_000_000, default_latency=0.005)
        compute = net.add_node("agg")

        for i in range(n_nodes):
            speed = slow_factor if i == 0 else 1.0
            node = net.add_node(f"s{i}", compute_speed=speed)
            node.store("text", generate_text(words_per_node))

        net.clear_logs()
        partials = []
        for i in range(n_nodes):
            r, _ = net.migrate_and_execute(word_count, f"s{i}", "text", "agg")
            partials.append(r)

        max_time = max(s.total_time for s in net.transfer_log)
        _, mt = compute.execute_local(merge_word_counts, partials)
        total = max_time + mt

        if baseline is None:
            baseline = total

        impact = total / baseline
        print(f"  {slow_factor:>10.1f}x | {total*1000:>14.2f} | {impact:>9.2f}x")


def run():
    """Run all Phase 4 experiments."""
    print("\n" + "="*80)
    print("  PHASE 4: Parallel Word Count with Auto-Migration")
    print("="*80)

    exp = run_parallel_wordcount()
    exp.print_report()

    run_skew_analysis()
    run_straggler_analysis()

    return exp


if __name__ == "__main__":
    run()
