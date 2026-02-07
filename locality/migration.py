"""
Code migration primitives: decorators and helpers for marking
functions as migratable and executing them at data locations.

This module provides the user-facing API:
- @migratable: marks a function as eligible for migration
- @migrate_to_data: always migrates to data location
- @remote_execute: explicitly targets a named node
"""

import functools
import cloudpickle
import inspect
import sys
from typing import Any, Callable, Optional


# Global registry: tracks the active NodeNetwork for decorated functions
_active_network = None
_active_compute_node = None


def set_runtime(network, compute_node: str):
    """Set the active network and default compute node for decorated functions."""
    global _active_network, _active_compute_node
    _active_network = network
    _active_compute_node = compute_node


def get_runtime():
    """Get the active (network, compute_node) tuple."""
    return _active_network, _active_compute_node


def code_size(func: Callable) -> int:
    """Measure the serialized size of a function in bytes."""
    return len(cloudpickle.dumps(func))


def serialize_function(func: Callable) -> bytes:
    """Serialize a function for transmission."""
    return cloudpickle.dumps(func)


def deserialize_function(data: bytes) -> Callable:
    """Deserialize a function received from network."""
    return cloudpickle.loads(data)


def migratable(func: Callable) -> Callable:
    """
    Decorator: marks a function as eligible for migration.

    The runtime will decide whether to migrate or fetch based on cost.
    The function's first argument should be a (node_name, data_key) tuple
    or actual data.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        network, compute_node = get_runtime()

        if network is None:
            # No network configured, run locally
            return func(*args, **kwargs)

        # Check if first arg is a DataRef
        if args and isinstance(args[0], DataRef):
            ref = args[0]
            result, stats = network.execute_with_decision(
                func, ref.node_name, ref.data_key, compute_node,
                None, *args[1:], **kwargs
            )
            wrapper._last_stats = stats
            return result

        # Otherwise run locally
        return func(*args, **kwargs)

    wrapper._last_stats = None
    wrapper._original = func
    wrapper._is_migratable = True
    return wrapper


def migrate_to_data(func: Callable) -> Callable:
    """
    Decorator: always migrates the function to the data location.
    No decision logic - always sends code to data.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        network, compute_node = get_runtime()

        if network is None:
            return func(*args, **kwargs)

        if args and isinstance(args[0], DataRef):
            ref = args[0]
            result, stats = network.migrate_and_execute(
                func, ref.node_name, ref.data_key, compute_node,
                *args[1:], **kwargs
            )
            wrapper._last_stats = stats
            return result

        return func(*args, **kwargs)

    wrapper._last_stats = None
    wrapper._original = func
    wrapper._is_migratable = True
    wrapper._always_migrate = True
    return wrapper


def remote_execute(node: str):
    """
    Decorator factory: explicitly targets a named node for execution.

    Usage:
        @remote_execute(node="server_a")
        def process(data):
            return len(data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            network, compute_node = get_runtime()

            if network is None:
                return func(*args, **kwargs)

            if args and isinstance(args[0], DataRef):
                ref = args[0]
                result, stats = network.migrate_and_execute(
                    func, node, ref.data_key, compute_node,
                    *args[1:], **kwargs
                )
                wrapper._last_stats = stats
                return result

            # If first arg is a data key string, use it directly
            if args and isinstance(args[0], str) and network.nodes[node].has_data(args[0]):
                data_key = args[0]
                result, stats = network.migrate_and_execute(
                    func, node, data_key, compute_node,
                    *args[1:], **kwargs
                )
                wrapper._last_stats = stats
                return result

            return func(*args, **kwargs)

        wrapper._last_stats = None
        wrapper._original = func
        wrapper._target_node = node
        return wrapper
    return decorator


class DataRef:
    """
    A reference to data on a specific node.

    Instead of holding the actual data, this is a lightweight pointer
    that the migration runtime uses to locate the data.
    """

    def __init__(self, node_name: str, data_key: str):
        self.node_name = node_name
        self.data_key = data_key

    def __repr__(self):
        return f"DataRef({self.node_name!r}, {self.data_key!r})"
