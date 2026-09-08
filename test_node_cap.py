"""test_node_cap.py — regression test for the node-cap bug.

The simulation's runaway guard and the validity check both used 6000, and
run_simulation appended the oversized frame *before* breaking. That made every
capped run fail is_valid with "too many nodes" — the safety brake was wired to
the reject button.

Run:  python test_node_cap.py
"""

from dg_engine import DEFAULT_PARAMS, is_valid, run_simulation

# A ring this big subdivides straight past a 100-node cap during start-up:
# 4 nodes at radius 20 -> 8 -> 16 -> 32 -> 64 -> 128, then edges fall under
# maxDistance and splitting stops. So the cap is guaranteed to trip.
GROWING = {**DEFAULT_PARAMS, "circle_radius": 20.0, "n_iterations": 60}
CAP = 100

failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


info = {}
snaps = run_simulation(GROWING, max_nodes=CAP, info=info)
final = snaps[-1]

check("no returned frame exceeds the cap",
      all(len(s) <= CAP for s in snaps),
      f"largest frame = {max(len(s) for s in snaps)} nodes, cap = {CAP}")

ok, reason = is_valid(final, max_nodes=CAP)
check("a capped run yields a valid shape", ok, reason)

check("truncation is reported, not silent",
      info.get("truncated") is True and info.get("stop_reason") == "node cap",
      f"info = {info}")

check("iterations_run matches the frames returned",
      info.get("iterations_run") == len(snaps) - 1,
      f"{info.get('iterations_run')} vs {len(snaps) - 1}")

# A run that finishes normally must not be flagged as truncated.
small = {**DEFAULT_PARAMS, "circle_radius": 5.0, "n_iterations": 30}
info2 = {}
snaps2 = run_simulation(small, max_nodes=6000, info=info2)
check("an uncapped run is not flagged truncated",
      info2.get("truncated") is False and info2.get("iterations_run") == 30,
      f"info = {info2}")

print()
print(f"{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
raise SystemExit(1 if failures else 0)
