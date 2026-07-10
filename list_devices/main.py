"""List video capture devices usable by the other remapy CLIs.

Probes camera indices with OpenCV (exactly as ``video_capture`` does) and prints
a summary of each device it can open and read. The reported index is the value to
pass to any other entry point via ``--source``::

    pixi run list-devices

    # then, e.g., use the reported index:
    pixi run capture --source 2
    pixi run pose --source 2

Run with ``--json`` for machine-readable output.
"""

import argparse
import json

from .devices import DeviceInfo, enumerate_devices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-index",
        type=int,
        default=9,
        help="Highest camera index to probe (default: 9). Linux /dev/videoN "
        "nodes above this are still included.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the device list as JSON instead of a human-readable summary.",
    )
    return parser.parse_args(argv)


def _format_device(dev: DeviceInfo) -> str:
    label = dev.name or "(unknown device)"
    parts = [f"  [{dev.index}] {label}"]
    if dev.node:
        parts.append(f"      node:       {dev.node}")
    parts.append(f"      resolution: {dev.width}x{dev.height}")
    if dev.fps > 0:
        parts.append(f"      fps:        {dev.fps:g}")
    parts.append(f"      backend:    {dev.backend}")
    parts.append(f"      use with:   --source {dev.source_arg}")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    devices = enumerate_devices(max_index=args.max_index)

    if args.json:
        print(json.dumps([vars(d) for d in devices], indent=2))
        return 0 if devices else 1

    if not devices:
        print("No compatible video capture devices found.")
        print("Check that a camera is connected and not in use by another program.")
        return 1

    count = len(devices)
    print(f"Found {count} compatible video capture device(s):\n")
    print("\n\n".join(_format_device(d) for d in devices))
    print(
        f"\nPass the bracketed index to any CLI, e.g. `pixi run capture --source {devices[0].index}`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
