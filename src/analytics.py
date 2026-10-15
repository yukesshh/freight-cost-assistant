"""Core shipment analytics, deterministic flags, and optional unsupervised detection."""
import pandas as pd
import numpy as np

ALIASES = {
    "origin": ["origin", "source", "from"],
    "destination": ["destination", "dest", "to"],
    "route_type": ["route_type", "route type", "route_category"],
    "quantity": ["quantity", "quantity_tonnes", "tonnes", "weight_tonnes", "weight"],
    "distance": ["distance", "distance_km", "distance (km)", "km"],
    "cost": ["cost", "freight_cost_inr", "freight_cost", "total_cost", "freight cost"],
    "date": ["shipment_date", "date", "ship_date", "shipment date"],
}

def _find_col(columns, logical):
    lower = {str(c).strip().lower(): c for c in columns}
    for alias in ALIASES[logical]:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None

def normalize_shipments(raw: pd.DataFrame, mapping=None) -> pd.DataFrame:
    """Normalize common CSV header variants to the internal schema."""
    if raw.empty:
        raise ValueError("The shipment file has no data rows.")
    mapping = mapping or {}
    result = {}
    for logical in ALIASES:
        col = mapping.get(logical) or _find_col(raw.columns, logical)
        if not col or col not in raw.columns:
            raise ValueError(f"Could not identify '{logical}' column. Select it in the upload form.")
        result[logical] = raw[col]
    df = pd.DataFrame(result)
    for c in ["origin", "destination", "route_type"]:
        df[c] = df[c].astype(str).str.strip()
        if df[c].isin(["", "nan", "None"]).any():
            raise ValueError(f"Column '{c}' contains blank values.")
    for c in ["quantity", "distance", "cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df[["quantity","distance","cost","date"]].isna().any().any():
        raise ValueError("The file has missing or invalid numeric/date values in required fields.")
    if (df["quantity"] <= 0).any() or (df["distance"] <= 0).any() or (df["cost"] < 0).any():
        raise ValueError("Quantity and distance must be positive; cost cannot be negative.")
    df["week_of"] = (df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")).dt.normalize()
    return df

def weekly_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    """Weekly rate = sum(freight cost) / sum(quantity * distance)."""
    df = shipments.copy()
    df["tonne_km"] = df["quantity"] * df["distance"]
    keys = ["origin", "destination", "route_type", "week_of"]
    weekly = df.groupby(keys, as_index=False).agg(
        total_cost=("cost", "sum"),
        total_tonne_km=("tonne_km", "sum"),
        shipment_count=("cost", "size")
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    return weekly.sort_values(keys).reset_index(drop=True)

def calculate_baselines(weekly: pd.DataFrame, history_window=8, threshold_pct=15.0) -> pd.DataFrame:
    """Historical baseline uses only prior observed route-weeks; peers exclude self."""
    df = weekly.sort_values(["origin","destination","route_type","week_of"]).copy().reset_index(drop=True)
    df["own_history_avg"] = np.nan
    df["history_weeks_used"] = 0
    for _, group in df.groupby(["origin","destination","route_type"], sort=False):
        indices = list(group.index)
        values = group["cost_per_tonne_km"].tolist()
        for pos, idx in enumerate(indices):
            prior = values[max(0, pos-history_window):pos]
            if prior:
                df.at[idx, "own_history_avg"] = float(np.mean(prior))
                df.at[idx, "history_weeks_used"] = len(prior)
    df["peer_avg"] = np.nan
    for idx, row in df.iterrows():
        peers = df[
            (df["week_of"] == row["week_of"]) &
            (df["route_type"] == row["route_type"]) &
            ~((df["origin"] == row["origin"]) & (df["destination"] == row["destination"]))
        ]["cost_per_tonne_km"]
        if len(peers):
            df.at[idx, "peer_avg"] = peers.mean()
    def pct(current, baseline):
        if pd.isna(baseline) or baseline == 0: return np.nan
        return (current-baseline)/baseline*100
    df["own_change_pct"] = [pct(a,b) for a,b in zip(df.cost_per_tonne_km, df.own_history_avg)]
    df["peer_change_pct"] = [pct(a,b) for a,b in zip(df.cost_per_tonne_km, df.peer_avg)]
    df["flag_rule"] = (
        (df["own_change_pct"].notna() & (df["own_change_pct"] >= threshold_pct)) |
        (df["peer_change_pct"].notna() & (df["peer_change_pct"] >= threshold_pct))
    )
    return df

def add_unsupervised_flags(df: pd.DataFrame, contamination=0.05) -> pd.DataFrame:
    """Flag statistical outliers using robust scaling with only pandas and NumPy.

    This is a lightweight alternative to Isolation Forest for environments where
    compiled scikit-learn/SciPy modules cannot be loaded. It ranks route-weeks by
    their average absolute robust z-score across rate and baseline-change features.
    """
    result = df.copy()
    feature_cols = ["cost_per_tonne_km", "own_change_pct", "peer_change_pct"]
    features = result[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    # Impute unavailable baseline deltas with zero and missing rates with median.
    rate_median = features["cost_per_tonne_km"].median()
    if pd.isna(rate_median):
        rate_median = 0.0
    features["cost_per_tonne_km"] = features["cost_per_tonne_km"].fillna(rate_median)
    features = features.fillna(0.0)

    if len(result) < 5:
        result["unsupervised_anomaly"] = False
        result["anomaly_score"] = np.nan
        return result

    # Median/MAD scaling reduces the influence of extreme values. Fall back to
    # IQR and then standard deviation when a feature has zero MAD.
    scores = []
    for col in feature_cols:
        values = features[col].astype(float)
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            q1, q3 = values.quantile([0.25, 0.75])
            scale = float((q3 - q1) / 1.349)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        scores.append(((values - median) / scale).abs().to_numpy())

    anomaly_score = np.mean(np.vstack(scores), axis=0)
    result["anomaly_score"] = anomaly_score
    fraction = min(max(float(contamination), 0.0), 1.0)
    count = max(1, int(np.ceil(len(result) * fraction))) if fraction > 0 else 0
    flags = np.zeros(len(result), dtype=bool)
    if count:
        # Stable sorting makes ties deterministic across runs.
        order = np.argsort(-anomaly_score, kind="stable")
        flags[order[:count]] = True
    result["unsupervised_anomaly"] = flags
    return result

def route_label(row):
    return f"{row['origin']}-{row['destination']}"

def format_output(df: pd.DataFrame, notes_found: dict, threshold_pct=15.0) -> pd.DataFrame:
    """Return columns exactly matching the supplied sample_output_format_v2.csv."""
    rows = []
    for _, r in df.iterrows():
        route = route_label(r)
        own = r["own_change_pct"]
        peer = r["peer_change_pct"]
        own_txt = f"{own:+.1f}% vs this route's past average" if pd.notna(own) else "N/A (no prior weeks)"
        peer_txt = f"{peer:+.1f}% vs similar-length routes this week" if pd.notna(peer) else "N/A (no peer routes)"
        info = notes_found.get((r["origin"], r["destination"], r["week_of"]), {})
        note_id = info.get("note_id", "")
        verdict = info.get("verdict", "unexplained" if r["flag_rule"] else "not_flagged")
        reason = info.get("reason", "")
        flagged = "Yes" if bool(r["flag_rule"]) else "No"
        if bool(r["flag_rule"]) and verdict == "justified":
            flagged = "No (justified)"
        if not reason:
            reason = ("Cost rise looks unexplained and worth a human review."
                      if bool(r["flag_rule"]) else "No anomaly threshold crossed based on available baselines.")
        rows.append({
            "route": route,
            "week_of": pd.Timestamp(r["week_of"]).strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(float(r["cost_per_tonne_km"]), 2),
            "vs_own_history": own_txt,
            "vs_similar_routes": peer_txt,
            "flagged": flagged,
            "matched_note_id": note_id,
            "reason": reason
        })
    return pd.DataFrame(rows, columns=[
        "route","week_of","cost_per_tonne_km","vs_own_history","vs_similar_routes",
        "flagged","matched_note_id","reason"
    ])
