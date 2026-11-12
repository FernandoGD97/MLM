"""Inspect canonical Parquets and optionally compare masking through the existing lab CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from lab.cpt.inspection.selection import TECHNIQUES


def call_cli(*, configuration: dict, path: Path) -> None:
    """Save the exact invocation as YAML and propagate the CLI's exit status."""
    path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "lab.cli", "run", str(path)], check=True)


def main() -> int:
    """Show statistics first, then run explicitly selected comparative masking techniques."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="+", help="One or more canonical Parquet files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", help="Fallback ISO 639-1 code, e.g. cs, es, en")
    parser.add_argument("--languages", nargs="+", help="One fallback language per corpus")
    parser.add_argument("--techniques", nargs="+", help="M0–M8, terms, names, or all")
    parser.add_argument("--terms", nargs="*", default=[])
    parser.add_argument(
        "--terms-file",
        "--terms-files",
        nargs="+",
        action="extend",
        default=[],
        help="TXT lists or Parquets with one terms column",
    )
    parser.add_argument("--probability", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--budget-mode", choices=("strict", "underfill"), default="strict")
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Opt in to printing text that may contain clinical PHI",
    )
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--base-model", help="Local encoder/tokenizer for entropy or attention")
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--attention-max-length", type=int, default=128)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    common = {
        "task": "cpt.inspect_corpora",
        "corpora": args.corpora,
        "language": args.language,
        "languages": args.languages,
    }
    call_cli(
        configuration={**common, "output_dir": str(root / "statistics")},
        path=root / "statistics.yaml",
    )
    techniques = args.techniques
    if techniques is None and sys.stdin.isatty():
        print("Available techniques:", ", ".join(TECHNIQUES), flush=True)
        answer = input("Techniques to preview (space-separated; Enter to finish): ").strip()
        techniques = answer.split() if answer else []
    if not techniques:
        print("Statistics complete. Use --techniques random terms ... to preview masking.")
        return 0
    term_files = list(args.terms_file)
    terms = []
    for value in args.terms:
        if Path(value).suffix.lower() in {".txt", ".parquet"} and Path(value).is_file():
            term_files.append(value)
        else:
            terms.append(value)
    if techniques == ["all"]:
        techniques = [
            name
            for name in TECHNIQUES
            if (name not in {"entropy", "attention"} or args.base_model)
            and (name != "terms" or terms or term_files)
        ]
        if not args.base_model:
            print("M8/attention omitted: supply --base-model for real model scoring.")
        if not terms and not term_files:
            print("Term masking omitted: supply --terms or --terms-file.")
    call_cli(
        configuration={
            **common,
            "output_dir": str(root / "masking"),
            "techniques": techniques,
            "terms": terms,
            "term_files": term_files,
            "attention_layers": args.attention_layers,
            "attention_max_length": args.attention_max_length,
            "probability": args.probability,
            "random_state": args.random_state,
            "case_sensitive": args.case_sensitive,
            "backfill": not args.no_backfill,
            "budget_mode": args.budget_mode,
            "show_examples": args.show_examples,
            "examples": args.examples,
            "base_model": args.base_model,
        },
        path=root / "masking.yaml",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
