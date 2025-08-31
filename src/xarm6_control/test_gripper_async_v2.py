#!/usr/bin/env python3
"""
Gripper socket end-to-end tester (client-side).

Run examples:
  python3 gripper_test.py --host 127.0.0.1 --port 22345 smoke
  python3 gripper_test.py bench --iters 50
  python3 gripper_test.py sweep --steps 11 --dwell 0.1 --tol 0.1
  python3 gripper_test.py fuzz --iters 100 --dwell 0.02
  python3 gripper_test.py watch --period 0.2

Notes:
- Assumes the async client lives in `gripper_client_async.py`. If the import fails,
  the script will tell you how to fix PYTHONPATH.
- If your server isn't publishing /gripper_position yet, GET may return None.
  Tests will keep going but will flag comparisons as "unknown".
"""

import argparse
import asyncio
import importlib
import random
import statistics
import sys
import time

# -------- import client (GripperClientAsync) --------
try:
    client_mod = importlib.import_module("gripper_client_async_v2")
    GripperClientAsync = getattr(client_mod, "GripperClientAsync")
except Exception as e:
    sys.stderr.write(
        "❌ Could not import GripperClientAsync from gripper_client_async_v2.py\n"
        f"Error: {e}\n"
        "Make sure gripper_client_async_v2.py is in the same directory or on PYTHONPATH.\n"
    )
    sys.exit(1)


# ---------------- helpers ----------------
async def ensure_connected(cli: GripperClientAsync, retries=30, delay=0.2):
    for i in range(retries):
        try:
            await cli.ping()
            return True
        except Exception as e:
            if i == 0:
                print(f"Waiting for server... ({e})")
            await asyncio.sleep(delay)
    return False


async def wait_for_position(cli: GripperClientAsync, timeout=3.0, period=0.1):
    """Wait until GET returns a non-None position or timeout."""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < timeout:
        try:
            resp = await cli.get()
            if isinstance(resp, dict) and resp.get("position") is not None:
                return resp
        except Exception:
            pass
        await asyncio.sleep(period)
    return None


def ok(resp):
    return isinstance(resp, dict) and resp.get("ok") is True


# ---------------- tests ----------------
async def test_smoke(cli: GripperClientAsync):
    print("== SMOKE ==")
    r = await cli.ping()
    print("PING:", r)

    for v in (0.0, 1.0, 0.5):
        rs = await cli.set(v)
        print(f"SET {v:.3f}:", rs)
        rg = await cli.get()
        print("GET:", rg)

    print("Smoke test complete.\n")


async def test_sweep(cli: GripperClientAsync, steps=11, dwell=0.05, tol=0.1):
    """
    Sweep normalized value 0..1 and compare measured position (if available).
    tol: acceptable absolute error (if position is available).
    """
    print(f"== SWEEP == steps={steps} dwell={dwell}s tol={tol}")
    # Ensure server is ready to report position (optional)
    first_pos = await wait_for_position(cli, timeout=2.0)
    if not first_pos or first_pos.get("position") is None:
        print("⚠️  Position not yet available; proceeding without comparisons.")

    for i in range(steps):
        target = i / (steps - 1) if steps > 1 else 0.0
        rs = await cli.set(target)
        await asyncio.sleep(dwell)
        rg = await cli.get()
        pos = rg.get("position") if isinstance(rg, dict) else None
        status = "unknown"
        if pos is not None:
            err = abs(pos - target)
            status = "OK" if err <= tol else f"Δ={err:.3f} (> tol)"
        print(f"SET {target:.3f} -> GET pos={pos}  [{status}]")
    print("Sweep complete.\n")


async def test_fuzz(cli: GripperClientAsync, iters=100, dwell=0.02):
    print(f"== FUZZ == iters={iters} dwell={dwell}s")
    errors = 0
    none_positions = 0
    for i in range(iters):
        v = random.random()
        try:
            rs = await cli.set(v)
            if not ok(rs):
                errors += 1
            await asyncio.sleep(dwell)
            rg = await cli.get()
            if not ok(rg):
                errors += 1
            if rg.get("position") is None:
                none_positions += 1
        except Exception:
            errors += 1
    print(f"Fuzz done. errors={errors}, GET with None position={none_positions}/{iters}\n")


async def test_bench(cli: GripperClientAsync, iters=50):
    print(f"== BENCH == iters={iters} (set+get roundtrip)")
    times = []
    values = [i / (iters - 1) if iters > 1 else 0.0 for i in range(iters)]
    # shuffle to avoid monotonic bias
    random.shuffle(values)

    for v in values:
        t0 = time.perf_counter()
        await cli.set(v)
        await cli.get()
        dt = time.perf_counter() - t0
        times.append(dt)

    mean = statistics.mean(times)
    p50 = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times)
    p99 = sorted(times)[int(0.99 * (len(times) - 1))]
    print(
        f"Latency (s): mean={mean:.4f}, median={p50:.4f}, p90={p90:.4f}, p99={p99:.4f}, "
        f"min={min(times):.4f}, max={max(times):.4f}\n"
    )


async def test_concurrent(cli_factory, concurrency=5, iters=20):
    """
    Launch multiple clients concurrently; each performs set->get loops.
    cli_factory: callable returning a new GripperClientAsync
    """
    print(f"== CONCURRENT == clients={concurrency} iters_per_client={iters}")

    async def worker(name):
        cli = cli_factory()
        await ensure_connected(cli)
        ok_count = 0
        try:
            for i in range(iters):
                v = random.random()
                r1 = await cli.set(v)
                r2 = await cli.get()
                if ok(r1) and ok(r2):
                    ok_count += 1
                await asyncio.sleep(0)  # yield
        finally:
            await cli.disconnect()
        print(f"  {name}: ok={ok_count}/{iters}")

    await asyncio.gather(*(worker(f"C{i}") for i in range(concurrency)))
    print("Concurrent test complete.\n")


# ---------------- main ----------------
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=22345)
    p.add_argument("--timeout", type=float, default=5.0, help="read timeout (s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke")
    s_bench = sub.add_parser("bench"); s_bench.add_argument("--iters", type=int, default=50)
    s_sweep = sub.add_parser("sweep")
    s_sweep.add_argument("--steps", type=int, default=11)
    s_sweep.add_argument("--dwell", type=float, default=0.05)
    s_sweep.add_argument("--tol", type=float, default=0.1)
    s_fuzz = sub.add_parser("fuzz")
    s_fuzz.add_argument("--iters", type=int, default=100)
    s_fuzz.add_argument("--dwell", type=float, default=0.02)
    s_conc = sub.add_parser("concurrent")
    s_conc.add_argument("--clients", type=int, default=5)
    s_conc.add_argument("--iters", type=int, default=20)
    s_watch = sub.add_parser("watch")
    s_watch.add_argument("--period", type=float, default=0.25)

    args = p.parse_args()

    # Single shared client for single-threaded tests
    cli = GripperClientAsync(args.host, args.port, args.timeout)

    # Wait for server up
    if not await ensure_connected(cli):
        print("❌ Could not connect to server (PING failed). Is it running?")
        return

    if args.cmd == "smoke":
        await test_smoke(cli)

    elif args.cmd == "bench":
        await test_bench(cli, iters=args.iters)

    elif args.cmd == "sweep":
        await test_sweep(cli, steps=args.steps, dwell=args.dwell, tol=args.tol)

    elif args.cmd == "fuzz":
        await test_fuzz(cli, iters=args.iters, dwell=args.dwell)

    elif args.cmd == "concurrent":
        # Use a factory so each worker has its own connection
        def mk():
            return GripperClientAsync(args.host, args.port, args.timeout)
        await test_concurrent(mk, concurrency=args.clients, iters=args.iters)

    elif args.cmd == "watch":
        print("== WATCH ==")
        try:
            while True:
                rg = await cli.get()
                print(rg)
                await asyncio.sleep(args.period)
        except KeyboardInterrupt:
            print("\nStopping watch.")

    await cli.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
