"""
Locality-Aware Compute: Code That Moves to Data

A framework for experimenting with computation migration -
instead of fetching data to code, we move code to data.
"""

from locality.node import Node, NodeNetwork
from locality.migration import migrate_to_data, migratable, remote_execute
from locality.decision import MigrationDecision, CostModel
from locality.parallel import auto_parallelize, migrate_and_parallelize

__all__ = [
    "Node",
    "NodeNetwork",
    "migrate_to_data",
    "migratable",
    "remote_execute",
    "MigrationDecision",
    "CostModel",
    "auto_parallelize",
    "migrate_and_parallelize",
]
