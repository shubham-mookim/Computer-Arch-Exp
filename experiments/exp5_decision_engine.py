"""
Experiment 5: Decision Engine Evaluation

Test the cost model's accuracy and the adaptive learner.

Questions:
  - How accurate is the cost model at predicting the right strategy?
  - Does the adaptive learner improve over time?
  - What's the crossover point for different function types?
"""

import random
import cloudpickle
from locality.node import NodeNetwork, _fmt_bytes
from locality.decision import MigrationDecision, AdaptiveDecision, CostModel
from locality.benchmark import ExperimentResult, BenchmarkResult


def identity_filter(data):
    """Pass-through: result is same size as input."""
    return data


def heavy_filter(data):
    """Aggressive filter: result is ~10% of input."""
    return [x for x in data if x > 0.9]


def aggregator(data):
    """Extreme reduction: returns a single number."""
    return sum(data)


def expander(data):
    """Expansion: result is larger than input (rare but happens)."""
    return [(x, x*2, x**2) for x in data]


FUNCTION_PROFILES = {
    "identity": (identity_filter, "~100%", "No reduction"),
    "light_filter": (lambda d: [x for x in d if x > 0.5], "~50%", "50% reduction"),
    "heavy_filter": (heavy_filter, "~10%", "90% reduction"),
    "aggregator": (aggregator, "~0%", "Single value result"),
    "expander": (expander, "~300%", "3x expansion"),
}


def run_crossover_analysis() -> ExperimentResult:
    """
    For each function profile, find the data size where migration wins.
    """
    experiment = ExperimentResult(
        name="Crossover Analysis",
        description="Finding the data size where migration beats fetching for different function types",
    )

    print(f"\n{'─'*80}")
    print(f"  Crossover Analysis: where migration wins for each function type")
    print(f"{'─'*80}")

    decision = MigrationDecision()
    bandwidth = 100_000_000
    latency = 0.005

    print(f"\n  {'Function':>15s} | {'Reduction':>10s} | {'Code Size':>10s} | "
          f"{'Crossover':>12s} | {'Description'}")
    print(f"  {'─'*15}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*30}")

    for name, (func, ratio_str, desc) in FUNCTION_PROFILES.items():
        code_bytes = len(cloudpickle.dumps(func))

        # Measure actual result size for a sample
        sample_data = [random.random() for _ in range(10_000)]
        result = func(sample_data)
        result_bytes = len(cloudpickle.dumps(result))
        data_bytes = len(cloudpickle.dumps(sample_data))

        # Scale result_bytes proportionally
        result_ratio = result_bytes / data_bytes

        # Find crossover: at what data size does migration win?
        crossover = decision.crossover_data_size(
            code_bytes, int(code_bytes * (1 + result_ratio)),
            bandwidth, latency
        )

        print(f"  {name:>15s} | {ratio_str:>10s} | {_fmt_bytes(code_bytes):>10s} | "
              f"{_fmt_bytes(int(crossover)):>12s} | {desc}")

    return experiment


def run_decision_accuracy():
    """
    Test: does the decision engine actually pick the faster strategy?
    Run both strategies, see if the prediction matches reality.
    """
    print(f"\n{'─'*80}")
    print(f"  Decision Accuracy: does the cost model predict correctly?")
    print(f"{'─'*80}")

    decision = MigrationDecision()
    correct = 0
    total = 0

    data_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    functions = [
        ("heavy_filter", heavy_filter),
        ("aggregator", aggregator),
        ("identity", identity_filter),
    ]

    print(f"\n  {'Function':>15s} | {'Data Size':>10s} | {'Predicted':>10s} | "
          f"{'Actual Winner':>14s} | {'Correct?':>8s}")
    print(f"  {'─'*15}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*14}─┼─{'─'*8}")

    for func_name, func in functions:
        for data_size in data_sizes:
            net = NodeNetwork(default_bandwidth=100_000_000, default_latency=0.005)
            data_node = net.add_node("data")
            compute_node = net.add_node("compute")

            data = [random.random() for _ in range(data_size)]
            stored_bytes = data_node.store("dataset", data)
            code_bytes = len(cloudpickle.dumps(func))

            # Predict
            result_sample = func(data[:min(1000, len(data))])
            estimated_result = len(cloudpickle.dumps(result_sample)) * (data_size / min(1000, data_size))

            predicted_migrate = decision.should_migrate(
                code_bytes, stored_bytes, int(estimated_result),
                100_000_000, 0.005
            )

            # Measure actual: fetch strategy
            net.clear_logs()
            d, fs = net.fetch_data("data", "dataset", "compute")
            r1, ct1 = compute_node.execute_local(func, d)
            fetch_time = fs.transfer_time + ct1

            # Measure actual: migrate strategy
            net.clear_logs()
            r2, ms = net.migrate_and_execute(func, "data", "dataset", "compute")
            migrate_time = ms.total_time

            actual_migrate_wins = migrate_time < fetch_time
            is_correct = predicted_migrate == actual_migrate_wins

            if is_correct:
                correct += 1
            total += 1

            pred_str = "migrate" if predicted_migrate else "fetch"
            actual_str = "migrate" if actual_migrate_wins else "fetch"
            mark = "YES" if is_correct else "NO"

            print(f"  {func_name:>15s} | {data_size:>10,d} | {pred_str:>10s} | "
                  f"{actual_str:>14s} | {mark:>8s}")

    accuracy = correct / total * 100
    print(f"\n  Overall accuracy: {correct}/{total} ({accuracy:.1f}%)")


def run_adaptive_learning():
    """
    Test the adaptive decision engine that learns from past executions.
    """
    print(f"\n{'─'*80}")
    print(f"  Adaptive Learning: does it improve over time?")
    print(f"{'─'*80}")

    adaptive = AdaptiveDecision()

    # Train on heavy_filter with various sizes
    training_data = [
        (1_000, heavy_filter),
        (5_000, heavy_filter),
        (10_000, heavy_filter),
        (50_000, heavy_filter),
    ]

    print(f"\n  Training phase:")
    for size, func in training_data:
        data = [random.random() for _ in range(size)]
        result = func(data)
        data_bytes = len(cloudpickle.dumps(data))
        result_bytes = len(cloudpickle.dumps(result))
        adaptive.record_observation("heavy_filter", data_bytes, result_bytes)
        ratio = result_bytes / data_bytes * 100
        print(f"    Observed: data={_fmt_bytes(data_bytes)}, result={_fmt_bytes(result_bytes)} "
              f"({ratio:.1f}% of input)")

    # Now predict for unseen sizes
    print(f"\n  Prediction phase:")
    print(f"  {'Data Size':>10s} | {'Predicted Result':>16s} | {'Actual Result':>14s} | {'Error':>8s}")
    print(f"  {'─'*10}─┼─{'─'*16}─┼─{'─'*14}─┼─{'─'*8}")

    for test_size in [2_000, 20_000, 100_000]:
        data = [random.random() for _ in range(test_size)]
        data_bytes = len(cloudpickle.dumps(data))

        predicted = adaptive.predict_result_size("heavy_filter", data_bytes)
        actual_result = heavy_filter(data)
        actual_bytes = len(cloudpickle.dumps(actual_result))

        error = abs(predicted - actual_bytes) / actual_bytes * 100

        print(f"  {test_size:>10,d} | {_fmt_bytes(predicted):>16s} | "
              f"{_fmt_bytes(actual_bytes):>14s} | {error:>7.1f}%")


def run():
    """Run all decision engine experiments."""
    print("\n" + "="*80)
    print("  PHASE 5: Decision Engine Evaluation")
    print("="*80)

    run_crossover_analysis()
    run_decision_accuracy()
    run_adaptive_learning()


if __name__ == "__main__":
    run()
