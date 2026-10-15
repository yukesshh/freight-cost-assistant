"""Context-note retrieval and optional LLM explanation grounded in matched evidence."""
import os
import pandas as pd

def normalize_notes(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["note_id","note_text","origin","destination","route_type","start_date","end_date"])
    cols = {str(c).strip().lower(): c for c in raw.columns}
    def choose(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    id_col = choose("note_id","id","note id","context_id")
    text_col = choose("note_text","text","note","description","context","details","reason")
    origin_col = choose("origin","source","from")
    dest_col = choose("destination","dest","to")
    type_col = choose("route_type","route type")
    start_col = choose("start_date","effective_from","date","note_date","valid_from")
    end_col = choose("end_date","effective_to","valid_to")
    if not text_col:
        # Allow free-form notes with any single text column.
        text_col = raw.columns[0]
    out = pd.DataFrame()
    out["note_id"] = raw[id_col].astype(str) if id_col else [f"NOTE-{i+1:04d}" for i in range(len(raw))]
    out["note_text"] = raw[text_col].fillna("").astype(str)
    out["origin"] = raw[origin_col].fillna("").astype(str).str.strip() if origin_col else ""
    out["destination"] = raw[dest_col].fillna("").astype(str).str.strip() if dest_col else ""
    out["route_type"] = raw[type_col].fillna("").astype(str).str.strip() if type_col else ""
    out["start_date"] = pd.to_datetime(raw[start_col], errors="coerce") if start_col else pd.NaT
    out["end_date"] = pd.to_datetime(raw[end_col], errors="coerce") if end_col else pd.NaT
    # If only a single note date is supplied, it applies on that date; if no dates exist, temporal fit is unknown.
    if start_col and not end_col:
        out["end_date"] = out["start_date"]
    return out

def retrieve_notes(row, notes: pd.DataFrame, max_notes=5):
    """Route/date filtering first; if note metadata is absent, return as context but not auto-justification."""
    if notes is None or notes.empty: return []
    week_start = pd.Timestamp(row["week_of"])
    week_end = week_start + pd.Timedelta(days=6)
    candidates = notes.copy()
    # If date metadata exists, require overlap; undated notes can be retrieved but are weak evidence.
    dated = candidates["start_date"].notna() & candidates["end_date"].notna()
    date_match = (~dated) | ((candidates["start_date"] <= week_end) & (candidates["end_date"] >= week_start))
    candidates = candidates[date_match]
    for col in ["origin","destination","route_type"]:
        candidates = candidates[(candidates[col] == "") | (candidates[col].str.casefold() == str(row[col]).casefold())]
    if candidates.empty: return []
    candidates["_specificity"] = sum((candidates[c] != "").astype(int) for c in ["origin","destination","route_type"])
    candidates = candidates.sort_values(["_specificity","note_id"], ascending=[False,True]).head(max_notes)
    return candidates.to_dict("records")

def _is_specific_and_dated(note, row):
    return bool(note.get("origin") and note.get("destination") and
                str(note["origin"]).casefold() == str(row["origin"]).casefold() and
                str(note["destination"]).casefold() == str(row["destination"]).casefold() and
                pd.notna(note.get("start_date")) and pd.notna(note.get("end_date")))

def generate_explanation(row, notes: pd.DataFrame, llm_enabled=False, model=None):
    """Returns (verdict, note_id, explanation, usage). Never let LLM decide numeric flag."""
    if not bool(row["flag_rule"]):
        return "not_flagged", "", "No anomaly threshold crossed based on available baselines.", {"input_tokens":0,"output_tokens":0,"calls":0}
    matches = retrieve_notes(row, notes)
    own = row.get("own_change_pct")
    peer = row.get("peer_change_pct")
    comps = []
    if pd.notna(own): comps.append(f"{own:+.1f}% vs this route's past average")
    if pd.notna(peer): comps.append(f"{peer:+.1f}% vs similar-length routes this week")
    summary = "; ".join(comps) or "cost increase against an available baseline"
    # Only a route-specific, date-specific note can be treated as potentially justifying evidence.
    evidence = [n for n in matches if _is_specific_and_dated(n, row)]
    note = evidence[0] if evidence else None
    if note is None:
        reason = f"No matching route-specific note with applicable dates was found. {summary}. Cost rise looks unexplained and worth a human review."
        return "unexplained", "", reason, {"input_tokens":0,"output_tokens":0,"calls":0}
    # Avoid using notes that explicitly indicate decreases/no effect as justification.
    note_text = str(note.get("note_text",""))
    lowered = note_text.lower()
    if any(term in lowered for term in ["discount", "decrease", "reduced", "no impact", "no effect", "stable demand", "did not affect"]):
        reason = f"The closest note {note['note_id']} does not support an increase: {note_text} {summary}. No genuine justification found; flagged for review."
        return "unexplained", "", reason, {"input_tokens":0,"output_tokens":0,"calls":0}

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not llm_enabled or not api_key:
        reason = f"Matches note {note['note_id']} for this route and date range: {note_text} The note is a plausible explanation, but it does not prove it caused the entire increase. {summary}."
        return "justified", str(note["note_id"]), reason, {"input_tokens":0,"output_tokens":0,"calls":0}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role":"system","content":(
                    "You explain freight cost anomalies for an audit. Use only the supplied note and numeric facts. "
                    "Do not claim the note caused the whole increase unless it says so. Never invent facts. "
                    "If evidence is insufficient, say so. Return 1-3 concise sentences in plain English."
                )},
                {"role":"user","content":(
                    f"Route: {row['origin']} to {row['destination']} ({row['route_type']}). "
                    f"Week starting: {pd.Timestamp(row['week_of']).date()}. "
                    f"Current cost per tonne-km: {row['cost_per_tonne_km']:.4f}. "
                    f"Historical change: {own if pd.notna(own) else 'unavailable'}. "
                    f"Peer change: {peer if pd.notna(peer) else 'unavailable'}. "
                    f"Verified candidate note ID: {note['note_id']}. Note text: {note_text}. "
                    "Write a cautious explanation, mention the note ID, and distinguish correlation from proof."
                )}
            ]
        )
        text = response.choices[0].message.content.strip()
        usage = response.usage
        return "justified", str(note["note_id"]), text, {
            "input_tokens": int(getattr(usage,"prompt_tokens",0) or 0),
            "output_tokens": int(getattr(usage,"completion_tokens",0) or 0),
            "calls": 1
        }
    except Exception as exc:
        # Safe deterministic fallback; a failed LLM call must not break the pipeline.
        reason = f"LLM explanation unavailable ({type(exc).__name__}); deterministic evidence summary: note {note['note_id']} says: {note_text} {summary}."
        return "justified", str(note["note_id"]), reason, {"input_tokens":0,"output_tokens":0,"calls":1,"error":type(exc).__name__}
