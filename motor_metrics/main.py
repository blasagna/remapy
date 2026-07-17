"""CLI: print (or export) the metric table for one or more recordings.

    python -m motor_metrics.main session.h5
    python -m motor_metrics.main session.h5 --exercise sit_hold
    python -m motor_metrics.main sessions/*.h5 --csv metrics.csv

For anything exploratory, prefer the notebook (``pixi run notebook``) — these numbers are
meant to be read as trends against Remy's own baseline, and a table in a terminal is a
poor place to see a trend.
"""

import argparse
import sys
from pathlib import Path

from recording.reader import Recording

from .quality import Gate
from .report import TRIAL_EXERCISES, metrics_table, session_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motor metrics for labeled trials in a recording."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="recording .h5 file(s)")
    parser.add_argument("--csv", type=Path, help="write the table to CSV instead of stdout")
    # Only the trial types, not the full label vocabulary: `calib` and `exclude` are
    # housekeeping and can never produce a row, so offering them here would just be a
    # way to ask for an empty table.
    parser.add_argument(
        "--exercise", choices=sorted(TRIAL_EXERCISES), help="only rows for this exercise"
    )
    parser.add_argument(
        "--min-visibility", type=float, default=Gate.min_visibility,
        help="landmark visibility gate (default: %(default)s)",
    )
    parser.add_argument(
        "--min-presence", type=float, default=Gate.min_presence,
        help="landmark presence gate (default: %(default)s)",
    )
    parser.add_argument(
        "--window-s", type=float,
        help="truncate every HOLD to this many seconds, so trials of unequal length are "
             "comparable on path length (transitions and crawls ignore it)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    gate = Gate(min_visibility=args.min_visibility, min_presence=args.min_presence)

    missing = [p for p in args.paths if not p.exists()]
    if missing:
        print(f"No such recording: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 1

    if len(args.paths) == 1:
        with Recording(args.paths[0]) as rec:
            table = metrics_table(
                rec, gate=gate, window_s=args.window_s, session=args.paths[0].stem
            )
    else:
        table = session_table(args.paths, gate=gate, window_s=args.window_s)

    if args.exercise and not table.empty:
        table = table[table["exercise"] == args.exercise]

    if table.empty:
        print(
            "No labeled trials found. Label some with `pixi run annotate <session.h5>` "
            "using the motor_metrics.labels vocabulary (e.g. `sit_hold;arms=free`).",
            file=sys.stderr,
        )
        return 1

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"Wrote {len(table)} rows to {args.csv}")
    else:
        import pandas as pd

        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
