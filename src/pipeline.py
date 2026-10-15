"""End-to-end processing shared by CLI and Streamlit app."""
from pathlib import Path
import json
import time
import pandas as pd
from src.analytics import normalize_shipments, weekly_costs, calculate_baselines, add_unsupervised_flags, format_output
from src.notes_llm import normalize_notes, generate_explanation

def run_pipeline(shipments_raw, notes_raw=None, mode="combined", threshold_pct=15.0,
                 contamination=0.05, llm_enabled=False, model=None):
    started = time.perf_counter()
    shipments = normalize_shipments(shipments_raw)
    weekly = weekly_costs(shipments)
    scored = calculate_baselines(weekly, history_window=8, threshold_pct=threshold_pct)
    if mode in ("Unsupervised only", "Combined"):
        scored = add_unsupervised_flags(scored, contamination=contamination)
    else:
        scored["unsupervised_anomaly"] = False
        scored["anomaly_score"] = float("nan")
    if mode == "Rule-based only":
        scored["flag_rule"] = scored["flag_rule"].astype(bool)
        scored["final_flag"] = scored["flag_rule"]
    elif mode == "Unsupervised only":
        scored["final_flag"] = scored["unsupervised_anomaly"]
    else:
        scored["final_flag"] = scored["flag_rule"] | scored["unsupervised_anomaly"]
    # Explanations are only attempted for final flagged rows. Preserve numeric rule flag for transparent comparison.
    notes = normalize_notes(notes_raw) if notes_raw is not None else normalize_notes(pd.DataFrame())
    notes_found, usage_total = {}, {"input_tokens":0,"output_tokens":0,"calls":0}
    for _, row in scored.iterrows():
        if not bool(row["final_flag"]):
            notes_found[(row.origin,row.destination,row.week_of)] = {
                "verdict":"not_flagged", "reason":"No anomaly threshold crossed by the selected detection mode.", "note_id":""
            }
            continue
        # LLM evidence/explanations are designed for rule-based cost rises; an unsupervised-only anomaly
        # may not represent a rise and should not be described as a price rise.
        if mode == "Unsupervised only" and not bool(row["flag_rule"]):
            notes_found[(row.origin,row.destination,row.week_of)] = {
                "verdict":"unexplained", "reason":"Unsupervised model marked this route-week as unusual. This is a statistical outlier, not necessarily a cost rise; review the metrics and notes.", "note_id":""
            }
            continue
        verdict, note_id, reason, usage = generate_explanation(row, notes, llm_enabled=llm_enabled, model=model)
        notes_found[(row.origin,row.destination,row.week_of)] = {"verdict":verdict,"note_id":note_id,"reason":reason}
        for k in ("input_tokens","output_tokens","calls"):
            usage_total[k] += int(usage.get(k,0))
    # Make output flags reflect the selected mode while retaining the sample's exact output schema.
    output = format_output(scored, notes_found, threshold_pct)
    if mode == "Unsupervised only":
        output["flagged"] = scored["final_flag"].map(lambda x: "Yes" if x else "No")
    elif mode == "Combined":
        output["flagged"] = [
            ("Yes" if bool(f) else "No") if not (bool(f) and v == "justified") else "No (justified)"
            for f,v in zip(scored["final_flag"], [notes_found[(r.origin,r.destination,r.week_of)]["verdict"] for _,r in scored.iterrows()])
        ]
    # If only unsupervised flagged it, make reason transparent.
    if mode in ("Unsupervised only","Combined"):
        for i, r in scored.iterrows():
            if bool(r["unsupervised_anomaly"]) and not bool(r["flag_rule"]):
                output.at[i,"reason"] = "Robust statistical outlier detection identified this route-week as unusual. This does not by itself establish a price rise or its cause."
                output.at[i,"matched_note_id"] = ""
    metrics = {
        "llm_enabled_requested": bool(llm_enabled),
        "llm_calls": usage_total["calls"],
        "input_tokens": usage_total["input_tokens"],
        "output_tokens": usage_total["output_tokens"],
        "estimated_cost_usd": None,
        "cost_note": "Set estimated_cost_usd using the selected model's current published rates; token counts are logged, rates can change.",
        "shipment_rows": int(len(shipments)),
        "route_week_rows": int(len(output)),
        "flagged_rows": int((scored["final_flag"]).sum()),
        "rule_flagged_rows": int(scored["flag_rule"].sum()),
        "unsupervised_flagged_rows": int(scored["unsupervised_anomaly"].sum()),
        "threshold_pct": float(threshold_pct),
        "history_window_weeks": 8,
        "unsupervised_contamination": float(contamination),
        "elapsed_seconds": round(time.perf_counter()-started, 4)
    }
    return output, scored, metrics
