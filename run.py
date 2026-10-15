"""Command-line runner. See README for environment setup."""
import argparse, json
from pathlib import Path
import pandas as pd
from src.pipeline import run_pipeline

def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--shipments", default=str(root/"data"/"shipment_records.csv"))
    p.add_argument("--notes", default=None)
    p.add_argument("--output", default=str(root/"output"/"results.csv"))
    p.add_argument("--mode", choices=["Combined","Rule-based only","Unsupervised only"], default="Combined")
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--llm", action="store_true")
    p.add_argument("--model", default=None)
    args = p.parse_args()
    ships = pd.read_csv(args.shipments)
    notes = pd.read_csv(args.notes) if args.notes else None
    output, scored, metrics = run_pipeline(ships, notes, args.mode, args.threshold, args.contamination, args.llm, args.model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    log = Path(args.output).with_name("run_metrics.json")
    log.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {len(output)} rows to {args.output}")
    print(f"Flagged: {metrics['flagged_rows']}; LLM calls: {metrics['llm_calls']}")
if __name__ == "__main__": main()
