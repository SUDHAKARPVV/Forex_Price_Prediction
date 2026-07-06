"""
Standalone report generator: turns an existing evaluation_report.json
(e.g. one from a previous run) into the human-readable HTML/PNG report,
without re-running training or evaluation.

Usage:
    python generate_report.py --input evaluation_report.json --output_dir report
"""
from __future__ import annotations

import argparse
import json

from utils.report import generate_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="evaluation_report.json")
    parser.add_argument("--output_dir", type=str, default="report")
    args = parser.parse_args()

    with open(args.input) as f:
        reports = json.load(f)

    html_path = generate_report(reports, output_dir=args.output_dir)
    print(f"Report written to {html_path}")
    print(f"Charts written to {args.output_dir}/charts/")
    print(f"Markdown summary written to {args.output_dir}/SUMMARY.md")
