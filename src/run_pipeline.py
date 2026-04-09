"""
run_pipeline.py

End-to-end orchestrator for the Amtrak ridership analysis pipeline.

Usage:
    python src/run_pipeline.py                  # run everything
    python src/run_pipeline.py --from train     # resume from train step onward
    python src/run_pipeline.py --only features  # run a single step
    python src/run_pipeline.py --dry-run        # print the plan without executing

Pipeline stages (in dependency order):
  1. parse_and_join   — join raw data → data/processed/stations.csv
  2. features         — add engineered features (metro_pop, nearby stations, etc.)
  3. build_gtfs       — add GTFS schedule features + NEC membership flag
  4. build_acs        — add ACS commute/income features (needs FCC API on first run)
  5. build_college    — add college enrollment proximity features
  6. build_tourism    — add overseas visitor features for top-50 tourist MSAs
  7. train            — EBM cross-validation + final model
  8. build_map        — generate underservice map HTML

Stages 2–5 all read/write data/processed/stations.csv in place.  They are
independent of each other (each adds its own columns) but must run serially
because they share the same output file.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# ── Pipeline definition ───────────────────────────────────────────────────────
# Each entry: (step_name, module_name, description)
PIPELINE = [
    ("parse_and_join",  "parse_and_join",          "Join raw data → stations.csv"),
    ("features",        "features",                "Engineered features (metro_pop, nearby stations)"),
    ("build_gtfs",      "build_gtfs_features",     "GTFS schedule features + NEC membership flag"),
    ("build_acs",       "build_acs_features",      "ACS commute mode + household income"),
    ("build_college",   "build_college_features",  "College enrollment proximity"),
    ("build_tourism",   "add_tourism_features",    "Overseas visitor features (top-50 tourist MSAs)"),
    ("build_candidates",  "build_candidates",              "Generate expansion candidate rows for top-100 cities without Amtrak"),
    ("train",            "train",                        "EBM cross-validation + final model"),
    ("predict_expansion", "predict_expansion_candidates", "Predict ridership for expansion candidates"),
    ("build_map",       "build_map",               "Generate underservice map HTML"),
]

STEP_NAMES = [name for name, _, _ in PIPELINE]


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s"


def run_step(module_name: str, step_name: str) -> None:
    """Import a module and call its main() function."""
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "main"):
        raise AttributeError(f"{module_name} has no main() function")
    mod.main()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the Amtrak ridership analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + " → ".join(STEP_NAMES),
    )
    parser.add_argument(
        "--from", dest="from_step", metavar="STEP",
        help=f"Resume from this step onward.  Choices: {', '.join(STEP_NAMES)}",
    )
    parser.add_argument(
        "--only", dest="only_step", metavar="STEP",
        help="Run a single step only.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the execution plan without running anything.",
    )
    args = parser.parse_args()

    # ── Resolve which steps to run ──
    if args.only_step:
        if args.only_step not in STEP_NAMES:
            parser.error(f"Unknown step '{args.only_step}'. Choices: {', '.join(STEP_NAMES)}")
        steps_to_run = [s for s in PIPELINE if s[0] == args.only_step]
    elif args.from_step:
        if args.from_step not in STEP_NAMES:
            parser.error(f"Unknown step '{args.from_step}'. Choices: {', '.join(STEP_NAMES)}")
        start_idx = STEP_NAMES.index(args.from_step)
        steps_to_run = PIPELINE[start_idx:]
    else:
        steps_to_run = PIPELINE

    # ── Print plan ──
    print("=" * 64)
    print("  Amtrak ridership pipeline")
    print("=" * 64)
    for i, (name, module, desc) in enumerate(steps_to_run, 1):
        print(f"  {i}. {name:20s} {desc}")
    print("=" * 64)

    if args.dry_run:
        print("\n  (dry run — nothing executed)")
        return

    # ── Execute ──
    t_pipeline = time.time()
    failed = False

    for i, (name, module, desc) in enumerate(steps_to_run, 1):
        print(f"\n{'─'*64}")
        print(f"  [{i}/{len(steps_to_run)}] {name}")
        print(f"{'─'*64}\n")

        t_step = time.time()
        try:
            run_step(module, name)
        except Exception as e:
            elapsed = fmt_elapsed(time.time() - t_step)
            print(f"\n  ✗ {name} FAILED after {elapsed}: {e}")
            print(f"\n  Pipeline stopped.  Fix the error and resume with:")
            print(f"    python src/run_pipeline.py --from {name}")
            failed = True
            break

        elapsed = fmt_elapsed(time.time() - t_step)
        print(f"\n  ✓ {name} completed in {elapsed}")

    total = fmt_elapsed(time.time() - t_pipeline)
    if not failed:
        print(f"\n{'='*64}")
        print(f"  Pipeline finished in {total}")
        print(f"{'='*64}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()