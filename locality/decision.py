"""
Decision engine: determines whether to migrate code to data
or fetch data to code, based on cost models.

The core insight: compare the total cost of both strategies
and pick the cheaper one. But "cost" has multiple dimensions:
network bytes, latency, compute utilization, etc.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CostModel:
    """
    Cost parameters for migration decisions.

    All costs are in abstract "cost units" - you can calibrate
    these to real dollars, seconds, or whatever metric you care about.
    """
    network_cost_per_byte: float = 1e-6       # cost per byte transferred
    network_cost_per_sec_latency: float = 10.0  # cost per second of latency
    compute_cost_per_sec: float = 1.0           # cost per second of compute

    def transfer_cost(self, size_bytes: int, bandwidth: float, latency: float) -> float:
        """Cost of transferring size_bytes over a link."""
        transfer_time = latency + (size_bytes / bandwidth)
        return (size_bytes * self.network_cost_per_byte +
                transfer_time * self.network_cost_per_sec_latency)

    def compute_cost(self, estimated_seconds: float) -> float:
        return estimated_seconds * self.compute_cost_per_sec


class MigrationDecision:
    """
    Decides whether code should migrate to data or data should be fetched.

    Strategy comparison:
    - Fetch data: transfer data to compute node, execute locally, done.
    - Migrate code: transfer code to data node, execute remotely, transfer result back.

    We pick whichever has lower estimated total cost.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self.history: list[dict] = []

    def should_migrate(self, code_size: int, data_size: int,
                       result_size: int, bandwidth: float,
                       latency: float) -> bool:
        """
        Decide: should code migrate to data?

        Returns True if migrating code is cheaper than fetching data.
        """
        cm = self.cost_model

        # Cost of fetching data to compute node
        cost_fetch = cm.transfer_cost(data_size, bandwidth, latency)

        # Cost of migrating code + returning result
        cost_migrate = (
            cm.transfer_cost(code_size, bandwidth, latency) +
            cm.transfer_cost(result_size, bandwidth, latency)
        )

        decision = cost_migrate < cost_fetch

        self.history.append({
            "code_size": code_size,
            "data_size": data_size,
            "result_size": result_size,
            "cost_fetch": cost_fetch,
            "cost_migrate": cost_migrate,
            "decision": "migrate" if decision else "fetch",
        })

        return decision

    def crossover_data_size(self, code_size: int, result_size: int,
                            bandwidth: float, latency: float) -> float:
        """
        Find the data size where migration becomes cheaper than fetching.

        This is the "crossover point" - below this size, fetching is fine;
        above it, migration wins.
        """
        # At crossover: cost_fetch == cost_migrate
        # network_cost_per_byte * data_size + latency_cost ==
        #   network_cost_per_byte * (code_size + result_size) + 2 * latency_cost
        #
        # Simplified (ignoring latency term differences):
        # data_size == code_size + result_size (approximately)
        #
        # More precisely:
        cm = self.cost_model
        migrate_fixed = (
            cm.transfer_cost(code_size, bandwidth, latency) +
            cm.transfer_cost(result_size, bandwidth, latency)
        )

        # cost_fetch(data_size) = cm.network_cost_per_byte * data_size + latency_term
        # We want cost_fetch(x) = migrate_fixed
        # x * cost_per_byte + latency + x/bandwidth * latency_cost = migrate_fixed
        per_byte_cost = cm.network_cost_per_byte + cm.network_cost_per_sec_latency / bandwidth
        latency_fixed = latency * cm.network_cost_per_sec_latency

        if per_byte_cost == 0:
            return float('inf')

        crossover = (migrate_fixed - latency_fixed) / per_byte_cost
        return max(0, crossover)

    def summary(self) -> dict:
        """Summarize decision history."""
        if not self.history:
            return {"total_decisions": 0}

        migrate_count = sum(1 for h in self.history if h["decision"] == "migrate")
        fetch_count = len(self.history) - migrate_count

        return {
            "total_decisions": len(self.history),
            "migrate_count": migrate_count,
            "fetch_count": fetch_count,
            "migrate_pct": migrate_count / len(self.history) * 100,
            "avg_data_size_migrate": (
                sum(h["data_size"] for h in self.history if h["decision"] == "migrate") / migrate_count
                if migrate_count else 0
            ),
            "avg_data_size_fetch": (
                sum(h["data_size"] for h in self.history if h["decision"] == "fetch") / fetch_count
                if fetch_count else 0
            ),
        }


class AdaptiveDecision(MigrationDecision):
    """
    Learns from past executions to improve migration decisions.

    Tracks actual result sizes from previous runs of the same function
    to make better predictions about future result sizes.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        super().__init__(cost_model)
        # function_name -> list of (data_size, result_size) observations
        self.observations: dict[str, list[tuple[int, int]]] = {}

    def record_observation(self, func_name: str, data_size: int, result_size: int):
        """Record an actual observation of data_size -> result_size."""
        if func_name not in self.observations:
            self.observations[func_name] = []
        self.observations[func_name].append((data_size, result_size))

    def predict_result_size(self, func_name: str, data_size: int) -> Optional[int]:
        """
        Predict result size based on past observations.
        Uses simple linear interpolation.
        """
        obs = self.observations.get(func_name, [])
        if not obs:
            return None

        if len(obs) == 1:
            # Single observation: assume same ratio
            ratio = obs[0][1] / max(obs[0][0], 1)
            return int(data_size * ratio)

        # Simple: average ratio
        ratios = [r / max(d, 1) for d, r in obs]
        avg_ratio = sum(ratios) / len(ratios)
        return int(data_size * avg_ratio)

    def should_migrate_adaptive(self, func_name: str, code_size: int,
                                data_size: int, bandwidth: float,
                                latency: float) -> bool:
        """
        Adaptive version of should_migrate that uses learned result sizes.
        """
        predicted = self.predict_result_size(func_name, data_size)
        if predicted is None:
            # No history: assume result is small (optimistic for migration)
            predicted = code_size

        return self.should_migrate(code_size, data_size, predicted, bandwidth, latency)
