import pandas as pd
from src.analytics import normalize_shipments, weekly_costs, calculate_baselines, add_unsupervised_flags

def sample():
    return pd.DataFrame({
        "origin":["A","A","A","B","B","B"],
        "destination":["B","B","B","C","C","C"],
        "route_type":["Short"]*6,
        "quantity_tonnes":[1]*6,
        "distance_km":[100]*6,
        "freight_cost_inr":[500,500,1000,400,400,400],
        "shipment_date":["2024-01-01","2024-01-08","2024-01-15","2024-01-01","2024-01-08","2024-01-15"]
    })

def test_normalizes_real_case_study_headers():
    df = normalize_shipments(sample())
    assert set(["origin","destination","quantity","distance","cost","week_of"]).issubset(df.columns)
    assert df.iloc[0]["week_of"] == pd.Timestamp("2024-01-01")

def test_weekly_weighted_cost():
    raw = sample().iloc[:2].copy()
    normalized = normalize_shipments(raw)
    # These records belong to different weeks, so each rate is 5.
    result = weekly_costs(normalized)
    assert all(abs(result["cost_per_tonne_km"] - 5.0) < 1e-8)

def test_historical_baseline_excludes_current_week():
    normalized = normalize_shipments(sample())
    weekly = weekly_costs(normalized)
    scored = calculate_baselines(weekly, history_window=8, threshold_pct=15)
    route = scored[(scored.origin=="A") & (scored.destination=="B")].sort_values("week_of")
    assert pd.isna(route.iloc[0]["own_history_avg"])
    assert route.iloc[1]["own_history_avg"] == 5.0
    assert route.iloc[2]["own_history_avg"] == 5.0
    assert bool(route.iloc[2]["flag_rule"])

def test_unsupervised_no_labels_required():
    normalized = normalize_shipments(sample())
    scored = calculate_baselines(weekly_costs(normalized))
    result = add_unsupervised_flags(scored, contamination=0.2)
    assert "unsupervised_anomaly" in result.columns
    assert len(result) == len(scored)
