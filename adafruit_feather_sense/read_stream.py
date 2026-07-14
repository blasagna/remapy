#!/usr/bin/env python3
"""Host-side reader for the Feather Sense USB serial telemetry stream.

Opens the board's USB CDC serial port, decodes the COBS-framed TLV records with
the shared :mod:`feather_protocol`, and prints each record as it arrives.

Usage (from the repo root, inside the pixi env)::

    pixi run python adafruit_feather_sense/read_stream.py            # auto-detect port
    pixi run python adafruit_feather_sense/read_stream.py --port /dev/ttyACM0
    pixi run python adafruit_feather_sense/read_stream.py --raw      # hexdump raw bytes (framing)
    pixi run python adafruit_feather_sense/read_stream.py --only accel,battery

The board writes binary only, but CircuitPython's console can emit startup text
on the same port; malformed leading bytes are skipped by the decoder and the
stream resyncs at the first frame delimiter.
"""

import argparse
import os
import sys
import time

import serial

# Import the shared modules living next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feather_protocol as fp  # noqa: E402
from stream import autodetect_port, to_si  # noqa: E402


def format_record(msg_type, timestamp_ms, values):
    name, fields = fp.MSG_INFO.get(msg_type, ("type0x%02x" % msg_type, ()))
    si = to_si(msg_type, values)
    parts = []
    for i, val in enumerate(si):
        label = fields[i] if i < len(fields) else "v%d" % i
        if isinstance(val, float):
            parts.append("%s=%.3f" % (label, val))
        else:
            parts.append("%s=%s" % (label, val))
    return "[%10d ms] %-9s %s" % (timestamp_ms, name, "  ".join(parts))


def format_error(timestamp_ms, values):
    source_type, message = values
    source = fp.MSG_INFO.get(source_type, ("0x%02x" % source_type,))[0]
    return "[%10d ms] %-9s source=%s  %s" % (timestamp_ms, "ERROR", source, message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial device (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (ignored by USB CDC, kept for pyserial)")
    parser.add_argument("--raw", action="store_true",
                        help="Hexdump raw incoming bytes instead of decoding (inspect COBS framing; "
                             "'00' bytes delimit frames and must not appear inside one)")
    parser.add_argument("--only",
                        help="Comma-separated stream names to show (e.g. accel,battery)")
    parser.add_argument("--stats", action="store_true",
                        help="Print per-stream sample rates once per second instead of records")
    args = parser.parse_args(argv)

    port = args.port or autodetect_port()
    if not port:
        parser.error("no serial port found; pass --port /dev/ttyACMx")

    only = set(args.only.split(",")) if args.only else None

    print("Reading %s ..." % port, file=sys.stderr)
    with serial.Serial(port, args.baud, timeout=0.1) as ser:
        decoder = fp.FrameDecoder()
        counts = {}
        last_report = time.monotonic()
        while True:
            chunk = ser.read(4096)
            if chunk and args.raw:
                print(chunk.hex(" "))
                continue
            if chunk:
                for msg_type, timestamp_ms, values in decoder.feed(chunk):
                    name = fp.MSG_INFO.get(msg_type, ("type0x%02x" % msg_type,))[0]
                    if only and name not in only:
                        continue
                    if args.stats:
                        counts[name] = counts.get(name, 0) + 1
                    elif msg_type == fp.MSG_ERROR:
                        print(format_error(timestamp_ms, values))
                    else:
                        print(format_record(msg_type, timestamp_ms, values))
            if args.stats:
                now = time.monotonic()
                if now - last_report >= 1.0:
                    summary = "  ".join("%s=%d/s" % (k, v) for k, v in sorted(counts.items()))
                    print("%-60s errors=%d" % (summary, decoder.errors))
                    counts.clear()
                    last_report = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
