#!/usr/bin/env python3
"""Host-side reader for the Feather Sense BLE telemetry stream (bleak).

Scans for the board advertising over the Nordic UART Service, connects, and
prints each decoded sensor record — the BLE counterpart of ``read_stream.py``.
Uses :class:`ble_stream.FeatherSenseBLEStream` (bleak under the hood), so the
records are already SI-converted.

Usage (from the repo root, inside the pixi env, with PC Bluetooth on)::

    pixi run python adafruit_feather_sense/read_ble.py                 # scan for "FeatherSense"
    pixi run python adafruit_feather_sense/read_ble.py --address AA:BB:CC:DD:EE:FF
    pixi run python adafruit_feather_sense/read_ble.py --stats         # per-stream rates
    pixi run python adafruit_feather_sense/read_ble.py --only accel,battery
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ble_stream import DEFAULT_NAME, FeatherSenseBLEStream  # noqa: E402
from stream import RateTracker  # noqa: E402


def format_record(rec):
    if rec.name == "error":
        source, message = rec.values
        return "[%10d ms] %-12s source=%s  %s" % (rec.timestamp_ms, "ERROR", source, message)
    parts = []
    for i, value in enumerate(rec.values):
        label = rec.fields[i] if i < len(rec.fields) else "v%d" % i
        if isinstance(value, float):
            parts.append("%s=%.3f" % (label, value))
        else:
            parts.append("%s=%s" % (label, value))
    return "[%10d ms] %-12s %s" % (rec.timestamp_ms, rec.name, "  ".join(parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default=None,
                        help="BLE address of the board (default: scan by name)")
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help='Advertised name to scan for (default: "%s")' % DEFAULT_NAME)
    parser.add_argument("--scan-timeout", type=float, default=6.0, help="Scan timeout seconds.")
    parser.add_argument("--only", help="Comma-separated stream names to show (e.g. accel,battery)")
    parser.add_argument("--stats", action="store_true",
                        help="Print per-stream sample rates once per second instead of records")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Stop after N seconds (default: run until Ctrl-C)")
    args = parser.parse_args(argv)

    only = set(args.only.split(",")) if args.only else None

    print("Scanning for %s ..." % (args.address or ('"%s"' % args.name)), file=sys.stderr)
    stream = FeatherSenseBLEStream.open_if_available(
        args.address, name=args.name, scan_timeout=args.scan_timeout
    )
    if stream is None:
        print("Feather Sense (BLE) not found or not streaming.", file=sys.stderr)
        return 1
    print("Connected: %s" % stream.port, file=sys.stderr)
    rates = RateTracker()
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while True:
            for rec in stream.poll():
                if only and rec.name not in only:
                    continue
                if args.stats:
                    rates.add(rec.name, rec.timestamp_ms)
                else:
                    print(format_record(rec))
            if args.stats and rates.due():
                print(rates.report(stream.errors), flush=True)
            if deadline and time.monotonic() >= deadline:
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
