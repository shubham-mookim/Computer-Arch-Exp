"""
Parallel execution: automatically split work across data locations.

When data is partitioned across multiple nodes, these primitives
detect the partitioning and fan out execution, then merge results.
"""

import functools
import time
import threading
import cloudpickle
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from locality.migration import get_runtime, DataRef


class PartitionedData:
    """
    Represents data that is split across multiple nodes.

    Each partition is a (node_name, data_key) pair.
    """

    def __init__(self, partitions: list[tuple[str, str]]):
        """
        Args:
            partitions: list of (node_name, data_key) tuples
        """
        self.partitions = partitions

    def __len__(self):
        return len(self.partitions)

    def __repr__(self):
        return f"PartitionedData({len(self.partitions)} partitions across {self.node_names})"

    @property
    def node_names(self) -> list[str]:
        return list(set(node for node, _ in self.partitions))


def auto_parallelize(merge_fn: Optional[Callable] = None):
    """
    Decorator factory: automatically parallelizes a function across
    partitioned data.

    The decorated function should accept data as its first argument.
    If the argument is a PartitionedData, the function is cloned to
    each partition, executed in parallel, and results are merged.

    Args:
        merge_fn: Function to merge results from all partitions.
                  Defaults to list concatenation.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args or not isinstance(args[0], PartitionedData):
                return func(*args, **kwargs)

            partitioned = args[0]
            remaining_args = args[1:]
            network, compute_node = get_runtime()

            if network is None:
                raise RuntimeError("No network configured. Call set_runtime() first.")

            results = []
            stats_list = []

            def run_on_partition(node_name, data_key):
                result, stats = network.migrate_and_execute(
                    func, node_name, data_key, compute_node,
                    *remaining_args, **kwargs
                )
                return result, stats

            # Execute in parallel across all partitions
            with ThreadPoolExecutor(max_workers=len(partitioned.partitions)) as executor:
                futures = {
                    executor.submit(run_on_partition, node, key): (node, key)
                    for node, key in partitioned.partitions
                }

                for future in as_completed(futures):
                    node, key = futures[future]
                    result, stats = future.result()
                    results.append((node, key, result))
                    stats_list.append(stats)

            # Sort by partition order to maintain determinism
            partition_order = {(n, k): i for i, (n, k) in enumerate(partitioned.partitions)}
            results.sort(key=lambda x: partition_order[(x[0], x[1])])
            just_results = [r for _, _, r in results]

            # Merge results
            if merge_fn is not None:
                merged = merge_fn(just_results)
            else:
                merged = just_results

            wrapper._last_stats = stats_list
            wrapper._last_results = just_results
            return merged

        wrapper._last_stats = None
        wrapper._last_results = None
        wrapper._original = func
        wrapper._is_parallel = True
        return wrapper
    return decorator


def migrate_and_parallelize(merge_fn: Optional[Callable] = None):
    """
    Decorator factory: combines migration + parallelization.
    Alias for auto_parallelize with explicit naming.
    """
    return auto_parallelize(merge_fn)


def merge_sum(results: list) -> Any:
    """Merge by summing all results."""
    return sum(results)


def merge_concat(results: list) -> list:
    """Merge by concatenating all lists."""
    out = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
        else:
            out.append(r)
    return out


def merge_dicts(results: list[dict]) -> dict:
    """Merge dictionaries by summing values for matching keys."""
    merged = {}
    for d in results:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged
