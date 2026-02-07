"""
Node simulation: each node has local storage, compute capability,
and can receive + execute migrated functions.

We simulate network latency and bandwidth between nodes to make
the experiments realistic without needing actual distributed hardware.
"""

import time
import sys
import threading
import cloudpickle
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TransferStats:
    """Tracks bytes transferred and timing for a single operation."""
    bytes_sent: int = 0
    bytes_received: int = 0
    transfer_time: float = 0.0
    compute_time: float = 0.0
    total_time: float = 0.0
    strategy: str = ""


@dataclass
class NetworkLink:
    """Simulated network link between two nodes."""
    bandwidth_bytes_per_sec: float = 100_000_000  # 100 MB/s default
    latency_sec: float = 0.001  # 1ms default

    def transfer_time(self, size_bytes: int) -> float:
        """Simulate transfer time for given payload size."""
        return self.latency_sec + (size_bytes / self.bandwidth_bytes_per_sec)


class Node:
    """
    A simulated compute node with local storage.

    Each node has:
    - A local key-value store (simulating local disk/memory)
    - Ability to execute arbitrary functions on local data
    - Simulated network links to other nodes
    - Metrics tracking for all operations
    """

    def __init__(self, name: str, compute_speed: float = 1.0):
        """
        Args:
            name: Unique identifier for this node
            compute_speed: Multiplier for compute time (1.0 = normal, 2.0 = 2x slower)
        """
        self.name = name
        self.compute_speed = compute_speed
        self.storage: dict[str, Any] = {}
        self._storage_sizes: dict[str, int] = {}  # cached serialized sizes
        self._lock = threading.Lock()
        self.execution_log: list[dict] = []

    def store(self, key: str, data: Any) -> int:
        """Store data locally. Returns size in bytes."""
        serialized = cloudpickle.dumps(data)
        size = len(serialized)
        with self._lock:
            self.storage[key] = data
            self._storage_sizes[key] = size
        return size

    def get(self, key: str) -> Any:
        """Retrieve data from local storage."""
        return self.storage.get(key)

    def data_size(self, key: str) -> int:
        """Get serialized size of stored data in bytes."""
        if key not in self._storage_sizes:
            if key in self.storage:
                self._storage_sizes[key] = len(cloudpickle.dumps(self.storage[key]))
            else:
                return 0
        return self._storage_sizes[key]

    def has_data(self, key: str) -> bool:
        return key in self.storage

    def keys(self) -> list[str]:
        return list(self.storage.keys())

    def total_data_size(self) -> int:
        """Total bytes of all stored data."""
        return sum(self._storage_sizes.get(k, 0) for k in self.storage)

    def execute_local(self, func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """
        Execute a function locally with timing.
        Returns (result, compute_time_seconds).
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * self.compute_speed
        # simulate slower compute by sleeping the difference
        if self.compute_speed > 1.0:
            extra = elapsed - (elapsed / self.compute_speed)
            time.sleep(extra)
        self.execution_log.append({
            "function": getattr(func, "__name__", str(func)),
            "compute_time": elapsed,
            "timestamp": time.time(),
        })
        return result, elapsed

    def execute_on_local_data(self, func: Callable, data_key: str,
                              *extra_args, **extra_kwargs) -> tuple[Any, float]:
        """
        Execute a function on data stored locally.
        The function receives the data as its first argument.
        """
        data = self.storage[data_key]
        return self.execute_local(func, data, *extra_args, **extra_kwargs)

    def __repr__(self):
        keys = list(self.storage.keys())
        total = self.total_data_size()
        return f"Node({self.name!r}, keys={keys}, total_data={_fmt_bytes(total)})"


class NodeNetwork:
    """
    A network of simulated nodes with configurable links.

    This is the core simulation environment: nodes hold data,
    and we can migrate code or fetch data between them, with
    realistic transfer costs.
    """

    def __init__(self, default_bandwidth: float = 100_000_000,
                 default_latency: float = 0.001):
        self.nodes: dict[str, Node] = {}
        self.links: dict[tuple[str, str], NetworkLink] = {}
        self.default_link = NetworkLink(default_bandwidth, default_latency)
        self.transfer_log: list[TransferStats] = []

    def add_node(self, name: str, compute_speed: float = 1.0) -> Node:
        node = Node(name, compute_speed)
        self.nodes[name] = node
        return node

    def get_node(self, name: str) -> Node:
        return self.nodes[name]

    def set_link(self, node_a: str, node_b: str,
                 bandwidth: float, latency: float):
        """Set network characteristics between two nodes (bidirectional)."""
        link = NetworkLink(bandwidth, latency)
        self.links[(node_a, node_b)] = link
        self.links[(node_b, node_a)] = link

    def get_link(self, src: str, dst: str) -> NetworkLink:
        return self.links.get((src, dst), self.default_link)

    def simulate_transfer(self, src: str, dst: str,
                          payload_bytes: int) -> float:
        """
        Simulate a network transfer between nodes.
        Returns the simulated transfer time in seconds.
        Actually sleeps to simulate the delay.
        """
        link = self.get_link(src, dst)
        delay = link.transfer_time(payload_bytes)
        time.sleep(delay)
        return delay

    def fetch_data(self, data_node: str, data_key: str,
                   compute_node: str) -> tuple[Any, TransferStats]:
        """
        Traditional approach: fetch data from data_node to compute_node.
        Returns (data, transfer_stats).
        """
        stats = TransferStats(strategy="fetch_data")
        node = self.nodes[data_node]
        data = node.get(data_key)
        data_bytes = node.data_size(data_key)

        # Simulate network transfer of data
        stats.bytes_sent = data_bytes
        stats.transfer_time = self.simulate_transfer(data_node, compute_node, data_bytes)
        stats.total_time = stats.transfer_time
        self.transfer_log.append(stats)
        return data, stats

    def migrate_and_execute(self, func: Callable, data_node: str,
                            data_key: str, compute_node: str,
                            *extra_args, **extra_kwargs) -> tuple[Any, TransferStats]:
        """
        Locality-aware approach: migrate code to data_node, execute there,
        return result to compute_node.

        Returns (result, transfer_stats).
        """
        stats = TransferStats(strategy="migrate_code")

        # Serialize the function (this is "the code" being migrated)
        code_bytes = cloudpickle.dumps(func)
        code_size = len(code_bytes)

        # Serialize extra args too (they travel with the code)
        args_bytes = cloudpickle.dumps(extra_args) if extra_args else b""
        kwargs_bytes = cloudpickle.dumps(extra_kwargs) if extra_kwargs else b""
        total_code_payload = code_size + len(args_bytes) + len(kwargs_bytes)

        # Simulate sending code to data node
        stats.bytes_sent = total_code_payload
        transfer_to = self.simulate_transfer(compute_node, data_node, total_code_payload)

        # Execute on data node
        target_node = self.nodes[data_node]
        result, compute_time = target_node.execute_on_local_data(
            func, data_key, *extra_args, **extra_kwargs
        )

        # Serialize result and send back
        result_bytes = cloudpickle.dumps(result)
        result_size = len(result_bytes)
        stats.bytes_received = result_size
        transfer_back = self.simulate_transfer(data_node, compute_node, result_size)

        stats.transfer_time = transfer_to + transfer_back
        stats.compute_time = compute_time
        stats.total_time = stats.transfer_time + stats.compute_time
        self.transfer_log.append(stats)
        return result, stats

    def execute_with_decision(self, func: Callable, data_node: str,
                              data_key: str, compute_node: str,
                              decision_engine: Optional[Any] = None,
                              *extra_args, **extra_kwargs) -> tuple[Any, TransferStats]:
        """
        Smart execution: let the decision engine choose migrate vs fetch.
        """
        from locality.decision import MigrationDecision

        if decision_engine is None:
            decision_engine = MigrationDecision()

        target = self.nodes[data_node]
        data_size = target.data_size(data_key)
        code_size = len(cloudpickle.dumps(func))

        # Estimate result size (pessimistic: assume same as data)
        # In practice we'd learn this over time
        estimated_result_size = code_size  # functions usually reduce data

        link = self.get_link(compute_node, data_node)

        should_migrate = decision_engine.should_migrate(
            code_size=code_size,
            data_size=data_size,
            result_size=estimated_result_size,
            bandwidth=link.bandwidth_bytes_per_sec,
            latency=link.latency_sec,
        )

        if should_migrate:
            return self.migrate_and_execute(
                func, data_node, data_key, compute_node,
                *extra_args, **extra_kwargs
            )
        else:
            # Fetch data, execute locally
            data, fetch_stats = self.fetch_data(data_node, data_key, compute_node)
            local_node = self.nodes[compute_node]
            result, compute_time = local_node.execute_local(func, data, *extra_args, **extra_kwargs)

            # Build combined stats
            result_size = len(cloudpickle.dumps(result))
            stats = TransferStats(
                strategy="fetch_data",
                bytes_sent=0,
                bytes_received=fetch_stats.bytes_sent,
                transfer_time=fetch_stats.transfer_time,
                compute_time=compute_time,
                total_time=fetch_stats.transfer_time + compute_time,
            )
            self.transfer_log.append(stats)
            return result, stats

    def summary(self) -> dict:
        """Summarize all transfers that have occurred."""
        total_bytes = sum(s.bytes_sent + s.bytes_received for s in self.transfer_log)
        total_time = sum(s.total_time for s in self.transfer_log)
        strategies = {}
        for s in self.transfer_log:
            strategies[s.strategy] = strategies.get(s.strategy, 0) + 1
        return {
            "total_transfers": len(self.transfer_log),
            "total_bytes_moved": total_bytes,
            "total_time": total_time,
            "strategies_used": strategies,
        }

    def clear_logs(self):
        self.transfer_log.clear()
        for node in self.nodes.values():
            node.execution_log.clear()


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
