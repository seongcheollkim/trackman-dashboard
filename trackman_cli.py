#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trackman_core import extract_activities, find_activity_by_date, load_trackman_json, write_outputs


def main() -> None:
    p = argparse.ArgumentParser(description="TrackMan JSON -> CSV converter")
    sub = p.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("convert", help="Convert getactivityreport JSON to CSV files")
    c1.add_argument("report_json")
    c1.add_argument("--out", default="outputs")
    c1.add_argument("--prefix", default=None)
    c1.add_argument("--raw", action="store_true", help="Use Measurement instead of NormalizedMeasurement")

    c2 = sub.add_parser("activities", help="Read Player Portal activities JSON and print sessions")
    c2.add_argument("activities_json")
    c2.add_argument("--date", help="YYYY-MM-DD. If provided, print matching reportLink only.")

    args = p.parse_args()

    if args.cmd == "convert":
        data = load_trackman_json(args.report_json)
        shots, summary = write_outputs(data, args.out, args.prefix, args.raw)
        print(f"Created: {shots}")
        print(f"Created: {summary}")
        return

    if args.cmd == "activities":
        data = load_trackman_json(args.activities_json)
        acts = extract_activities(data)
        if args.date:
            act = find_activity_by_date(acts, args.date)
            if not act:
                raise SystemExit(f"No activity found for {args.date}")
            print(json.dumps(act, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(acts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
