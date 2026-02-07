"""
Experiment 3: Distributed Join with Data Gravity

The spicy problem: your function needs data from MULTIPLE locations.

Scenario:
  - Server A has user profiles (small-ish)
  - Server B has user events (large)
  - Server C has user purchases (medium)
  - We want to join them for analysis

Questions:
  - Does the heaviest data always win?
  - When is replication of small data worth it?
  - How do different data size ratios change the decision?
"""

import random
import cloudpickle
from locality.node import NodeNetwork, _fmt_bytes
from locality.benchmark import BenchmarkResult, ExperimentResult


def generate_users(n: int) -> list[dict]:
    """Generate user profiles."""
    return [{"id": i, "name": f"user_{i}", "tier": random.choice(["free", "pro", "enterprise"])}
            for i in range(n)]


def generate_events(n_users: int, events_per_user: int) -> list[dict]:
    """Generate user events (the big one)."""
    events = []
    for uid in range(n_users):
        for _ in range(events_per_user):
            events.append({
                "user_id": uid,
                "type": random.choice(["click", "view", "scroll", "submit", "search"]),
                "value": random.randint(1, 1000),
            })
    return events


def generate_purchases(n_users: int, purchases_per_user: int) -> list[dict]:
    """Generate purchase records."""
    purchases = []
    for uid in range(n_users):
        for _ in range(random.randint(0, purchases_per_user)):
            purchases.append({
                "user_id": uid,
                "amount": round(random.uniform(5.0, 500.0), 2),
                "category": random.choice(["electronics", "books", "clothing", "food"]),
            })
    return purchases


def join_and_analyze(users, events, purchases):
    """
    Join all three datasets and compute per-user analytics.
    This is the function we need to decide WHERE to run.
    """
    # Index events and purchases by user_id
    events_by_user = {}
    for e in events:
        uid = e["user_id"]
        if uid not in events_by_user:
            events_by_user[uid] = []
        events_by_user[uid].append(e)

    purchases_by_user = {}
    for p in purchases:
        uid = p["user_id"]
        if uid not in purchases_by_user:
            purchases_by_user[uid] = []
        purchases_by_user[uid].append(p)

    # Join and analyze
    results = []
    for user in users:
        uid = user["id"]
        user_events = events_by_user.get(uid, [])
        user_purchases = purchases_by_user.get(uid, [])
        results.append({
            "user_id": uid,
            "tier": user["tier"],
            "event_count": len(user_events),
            "total_event_value": sum(e["value"] for e in user_events),
            "purchase_count": len(user_purchases),
            "total_spent": sum(p["amount"] for p in user_purchases),
        })
    return results


def run_data_gravity(n_users: int = 1_000,
                     events_per_user: int = 50,
                     purchases_per_user: int = 5,
                     bandwidth: float = 100_000_000,
                     latency: float = 0.005) -> ExperimentResult:
    """
    Compare different strategies for multi-source join.
    """
    net = NodeNetwork(default_bandwidth=bandwidth, default_latency=latency)

    # Create nodes
    profiles_node = net.add_node("profiles_server")
    events_node = net.add_node("events_server")
    purchases_node = net.add_node("purchases_server")
    compute_node = net.add_node("compute")

    # Generate and store data
    users = generate_users(n_users)
    events = generate_events(n_users, events_per_user)
    purchases = generate_purchases(n_users, purchases_per_user)

    profiles_size = profiles_node.store("users", users)
    events_size = events_node.store("events", events)
    purchases_size = purchases_node.store("purchases", purchases)

    experiment = ExperimentResult(
        name="Data Gravity: Multi-Source Join",
        description=(
            f"Join {n_users:,} users x {events_per_user} events x ~{purchases_per_user} purchases. "
            f"Profiles={_fmt_bytes(profiles_size)}, Events={_fmt_bytes(events_size)}, "
            f"Purchases={_fmt_bytes(purchases_size)}"
        ),
        metadata={
            "profiles_bytes": profiles_size,
            "events_bytes": events_size,
            "purchases_bytes": purchases_size,
            "total_data_bytes": profiles_size + events_size + purchases_size,
        }
    )

    # Strategy 1: Move everything to compute node
    net.clear_logs()
    u, s1 = net.fetch_data("profiles_server", "users", "compute")
    e, s2 = net.fetch_data("events_server", "events", "compute")
    p, s3 = net.fetch_data("purchases_server", "purchases", "compute")
    result_central, ct = compute_node.execute_local(join_and_analyze, u, e, p)

    fetch_all_time = s1.transfer_time + s2.transfer_time + s3.transfer_time + ct
    fetch_all_bytes = s1.bytes_sent + s2.bytes_sent + s3.bytes_sent

    experiment.add(BenchmarkResult(
        name="fetch_all",
        strategy="Fetch All to Compute",
        total_time=fetch_all_time,
        transfer_time=s1.transfer_time + s2.transfer_time + s3.transfer_time,
        compute_time=ct,
        bytes_transferred=fetch_all_bytes,
        result_value=len(result_central),
    ))

    # Strategy 2: Move to heaviest data (events server)
    # Send users + purchases to events_server, execute there
    net.clear_logs()
    # Transfer profiles and purchases to events server
    u2, su = net.fetch_data("profiles_server", "users", "events_server")
    p2, sp = net.fetch_data("purchases_server", "purchases", "events_server")

    # Execute on events server (events are already local)
    e_local = events_node.get("events")
    result_gravity, ct2 = events_node.execute_local(join_and_analyze, u2, e_local, p2)

    # Send result back to compute
    result_bytes = len(cloudpickle.dumps(result_gravity))
    result_transfer = net.simulate_transfer("events_server", "compute", result_bytes)

    gravity_transfer_time = su.transfer_time + sp.transfer_time + result_transfer
    gravity_bytes = su.bytes_sent + sp.bytes_sent + result_bytes

    experiment.add(BenchmarkResult(
        name="gravity",
        strategy="Gravity (go to events)",
        total_time=gravity_transfer_time + ct2,
        transfer_time=gravity_transfer_time,
        compute_time=ct2,
        bytes_transferred=gravity_bytes,
        result_value=len(result_gravity),
    ))

    # Strategy 3: Smart - move two small datasets to the large one
    # Same as gravity in this case but let's compute the optimal choice
    net.clear_logs()
    sizes = {
        "profiles_server": profiles_size,
        "events_server": events_size,
        "purchases_server": purchases_size,
    }
    # Pick the node with the most data
    heaviest = max(sizes, key=sizes.get)
    others = [n for n in sizes if n != heaviest]
    other_total = sum(sizes[n] for n in others)

    print(f"\n  Data gravity analysis:")
    print(f"    Heaviest node: {heaviest} ({_fmt_bytes(sizes[heaviest])})")
    print(f"    Others total:  {_fmt_bytes(other_total)}")
    print(f"    Moving {_fmt_bytes(other_total)} instead of {_fmt_bytes(sizes[heaviest])}")
    print(f"    Savings: {_fmt_bytes(sizes[heaviest] - other_total)} "
          f"({(1 - other_total/sizes[heaviest])*100:.0f}% less)")

    # Verify correctness
    assert len(result_central) == len(result_gravity), "Results differ!"

    return experiment


def run_ratio_sweep():
    """
    How do different data size ratios affect which strategy wins?
    """
    print(f"\n{'─'*70}")
    print(f"  Data Ratio Sweep: when does gravity matter?")
    print(f"{'─'*70}")
    print(f"  {'Events/User':>12s} | {'Events Size':>12s} | {'Fetch All (ms)':>14s} | "
          f"{'Gravity (ms)':>14s} | {'Savings':>10s}")
    print(f"  {'─'*12}─┼─{'─'*12}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*10}")

    for events_per_user in [5, 10, 50, 100, 200]:
        net = NodeNetwork(default_bandwidth=100_000_000, default_latency=0.005)
        profiles_node = net.add_node("profiles")
        events_node = net.add_node("events")
        purchases_node = net.add_node("purchases")
        compute_node = net.add_node("compute")

        n_users = 1_000
        users = generate_users(n_users)
        events = generate_events(n_users, events_per_user)
        purchases = generate_purchases(n_users, 5)

        ps = profiles_node.store("users", users)
        es = events_node.store("events", events)
        prs = purchases_node.store("purchases", purchases)

        # Fetch all
        net.clear_logs()
        u, s1 = net.fetch_data("profiles", "users", "compute")
        e, s2 = net.fetch_data("events", "events", "compute")
        p, s3 = net.fetch_data("purchases", "purchases", "compute")
        _, ct = compute_node.execute_local(join_and_analyze, u, e, p)
        fetch_time = s1.transfer_time + s2.transfer_time + s3.transfer_time + ct

        # Gravity: move to events node
        net.clear_logs()
        u2, su = net.fetch_data("profiles", "users", "events")
        p2, sp = net.fetch_data("purchases", "purchases", "events")
        e_local = events_node.get("events")
        _, ct2 = events_node.execute_local(join_and_analyze, u2, e_local, p2)
        rb = len(cloudpickle.dumps(_))
        rt = net.simulate_transfer("events", "compute", rb)
        gravity_time = su.transfer_time + sp.transfer_time + rt + ct2

        savings = (1 - gravity_time / fetch_time) * 100

        print(f"  {events_per_user:>12d} | {_fmt_bytes(es):>12s} | {fetch_time*1000:>14.2f} | "
              f"{gravity_time*1000:>14.2f} | {savings:>9.1f}%")


def run():
    """Run all Phase 3 experiments."""
    print("\n" + "="*80)
    print("  PHASE 3: Distributed Join with Data Gravity")
    print("="*80)

    exp = run_data_gravity()
    exp.print_report()

    run_ratio_sweep()

    return exp


if __name__ == "__main__":
    run()
