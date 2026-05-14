"""
build_graph_features.py

Build a directed weighted graph from the Amtrak GTFS feed (edges = trips/week
between station pairs along the same trip) and compute node-level graph
features per Amtrak station.  Joins the result to data/processed/stations.csv
in-place.

Features added:
  - dest_metro_pop_weighted_sum
      For each station, sum over all destinations of:
          (trips/week from station to destination) * (destination's metro_pop)
      High for hub stations whose trains reach large metros frequently.
  - n_distinct_destinations
      Number of distinct station codes reachable via a single trip
      (any number of intermediate stops).  Measures connectivity breadth.
  - eigenvector_centrality
      Standard eigenvector centrality on the undirected weighted graph
      (edge weight = combined trips/week in both directions).  High for
      stations connected to other well-connected stations.

Edge construction: ALL-pairs along each trip.  If a trip stops at A -> B -> C -> D,
edges (A,B), (A,C), (A,D), (B,C), (B,D), (C,D) all get +days_per_week of weight.
Reasoning: a passenger boarding at A to ride to D contributes to "demand from A
toward D-class destinations," not just to the first hop.

Reuses the station-matching logic from build_gtfs_features.py so trip stop_ids
get mapped to the same station codes used elsewhere in the dataset.

Run directly:
    python src/build_graph_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from build_gtfs_features import (  # noqa: E402  -- intentional sys.path setup above
    load_gtfs,
    compute_per_stop_features,
    match_stops_to_stations,
)

PROCESSED = ROOT / "data" / "processed"

GRAPH_FEATURE_COLS = [
    "dest_metro_pop_weighted_sum",
    "n_distinct_destinations",
    "eigenvector_centrality",
]


# -- Edge construction ---------------------------------------------------------

def build_edge_list(st_filt: pd.DataFrame, trips_filt: pd.DataFrame,
                    stop_to_code: dict) -> pd.DataFrame:
    """
    Build a directed edge list from GTFS stop_times.

    For each trip, enumerate all ordered station-code pairs along the trip
    (consecutive AND non-consecutive) and weight each by the trip pattern's
    days_per_week.  Returns a DataFrame with columns:
        src_code, dst_code, weight

    where weight is trips/week between src and dst on this trip pattern.
    Aggregation across trip patterns happens downstream.
    """
    # Merge days_per_week onto stop_times
    st = st_filt.merge(
        trips_filt[["trip_id", "days_per_week"]],
        on="trip_id", how="left",
    )
    st["days_per_week"] = st["days_per_week"].fillna(0)

    # Map stop_id -> matched station code; drop rows that didn't match
    st["code"] = st["stop_id"].map(stop_to_code)
    st = st.dropna(subset=["code"])

    # Sort within each trip by stop_sequence so we iterate trips in order
    st = st.sort_values(["trip_id", "stop_sequence"])

    edges = []
    # Group once, iterate once. A few thousand trips total -- fine in Python.
    for trip_id, grp in st.groupby("trip_id", sort=False):
        codes = grp["code"].tolist()
        # days_per_week is the same across rows in a trip (it's a trip attribute)
        dpw = grp["days_per_week"].iloc[0]
        if dpw <= 0 or len(codes) < 2:
            continue
        n = len(codes)
        # All ordered pairs (i, j) with i < j -> directed src->dst
        # For symmetric features we'll combine both directions later.
        for i in range(n):
            for j in range(i + 1, n):
                if codes[i] == codes[j]:
                    continue  # self-edge from a station hit twice on a trip
                edges.append((codes[i], codes[j], dpw))

    edge_df = pd.DataFrame(edges, columns=["src_code", "dst_code", "weight"])

    # Aggregate over trip patterns: total trips/week between each ordered pair
    edge_df = (
        edge_df.groupby(["src_code", "dst_code"], as_index=False)["weight"].sum()
    )
    print(f"  Built edge list: {len(edge_df)} directed (src, dst) pairs")
    return edge_df


# -- Node features -------------------------------------------------------------

def compute_node_features(edge_df: pd.DataFrame,
                          stations: pd.DataFrame) -> pd.DataFrame:
    """
    Given an edge list and stations metadata, compute per-station graph features.
    Returns a DataFrame with columns: code + GRAPH_FEATURE_COLS.
    """
    # Look up metro_pop per code
    pop_map = stations.set_index("code")["metro_pop"].to_dict()

    # -- 1. Weighted destination metro_pop sum --
    # For each src, sum over dst of (weight * metro_pop[dst]).
    # Missing metro_pop treated as 0 (won't contribute).
    edge_df = edge_df.copy()
    edge_df["dst_pop"] = edge_df["dst_code"].map(pop_map).fillna(0)
    edge_df["pop_weighted"] = edge_df["weight"] * edge_df["dst_pop"]

    by_src = edge_df.groupby("src_code").agg(
        dest_metro_pop_weighted_sum=("pop_weighted", "sum"),
        n_distinct_destinations=("dst_code", "nunique"),
    ).reset_index()
    by_src = by_src.rename(columns={"src_code": "code"})

    # -- 2. Eigenvector centrality on undirected graph --
    # Combine src->dst and dst->src weights into a symmetric weighted adjacency.
    sym = edge_df[["src_code", "dst_code", "weight"]].copy()
    sym_rev = sym.rename(columns={"src_code": "dst_code", "dst_code": "src_code"})
    sym_both = pd.concat([sym, sym_rev], ignore_index=True)
    sym_agg = sym_both.groupby(["src_code", "dst_code"], as_index=False)["weight"].sum()
    # sym_agg now has each unordered pair twice: (A,B) and (B,A) with equal weight.

    # Build adjacency matrix
    codes = sorted(set(sym_agg["src_code"]).union(set(sym_agg["dst_code"])))
    code_to_idx = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    A = np.zeros((n, n), dtype=np.float64)
    # Vectorised fill instead of per-row loop
    src_idx = sym_agg["src_code"].map(code_to_idx).values
    dst_idx = sym_agg["dst_code"].map(code_to_idx).values
    A[src_idx, dst_idx] = sym_agg["weight"].values

    # Eigenvector centrality = entries of the eigenvector of the largest
    # eigenvalue of A.  For a symmetric nonneg matrix, this is real & nonneg
    # (Perron-Frobenius for the dominant component).  Use eigh for stability.
    # Note: if the graph is disconnected, eigh still returns sensible centralities
    # within the dominant component; isolated/peripheral nodes get ~0.
    eigvals, eigvecs = np.linalg.eigh(A)
    # eigh returns ascending -- take the last column (largest eigenvalue)
    dominant_vec = eigvecs[:, -1]
    # Sign convention: make the dominant entry positive
    if dominant_vec[np.argmax(np.abs(dominant_vec))] < 0:
        dominant_vec = -dominant_vec
    # Normalize to max=1 for readability (any positive scaling is valid)
    if dominant_vec.max() > 0:
        dominant_vec = dominant_vec / dominant_vec.max()

    centrality_df = pd.DataFrame({
        "code": codes,
        "eigenvector_centrality": dominant_vec,
    })

    # -- Merge node features --
    feat = by_src.merge(centrality_df, on="code", how="outer")
    # Fill NaNs: stations with no outbound edges get 0 for sum & count;
    # stations not in the graph at all get 0 for centrality.
    feat["dest_metro_pop_weighted_sum"] = feat["dest_metro_pop_weighted_sum"].fillna(0)
    feat["n_distinct_destinations"] = feat["n_distinct_destinations"].fillna(0).astype(int)
    feat["eigenvector_centrality"] = feat["eigenvector_centrality"].fillna(0)

    print(f"  Computed graph features for {len(feat)} stations")
    print(f"\nTop 10 by dest_metro_pop_weighted_sum (scaled to millions of pop-trips):")
    top = feat.nlargest(10, "dest_metro_pop_weighted_sum")[
        ["code", "dest_metro_pop_weighted_sum",
         "n_distinct_destinations", "eigenvector_centrality"]
    ].copy()
    top["dest_metro_pop_weighted_sum"] = (
        top["dest_metro_pop_weighted_sum"] / 1e6
    ).round(1)
    top["eigenvector_centrality"] = top["eigenvector_centrality"].round(3)
    print(top.to_string(index=False))

    return feat


# -- Join to stations ----------------------------------------------------------

def join_to_stations(graph_feats: pd.DataFrame,
                     stations: pd.DataFrame) -> pd.DataFrame:
    """Left-join graph features onto stations.csv; drop old graph cols first."""
    for col in GRAPH_FEATURE_COLS:
        if col in stations.columns:
            stations = stations.drop(columns=[col])

    merged = stations.merge(graph_feats, on="code", how="left")

    # Fill NaNs for stations not present in graph (the 44 GTFS-missing ones):
    # 0 for all three features -- they have no graph presence.
    for col in GRAPH_FEATURE_COLS:
        if col == "n_distinct_destinations":
            merged[col] = merged[col].fillna(0).astype(int)
        else:
            merged[col] = merged[col].fillna(0)

    filled = merged[GRAPH_FEATURE_COLS[0]].gt(0).sum()
    print(f"\nJoined graph features to {filled}/{len(merged)} stations "
          f"(others have 0 -- no GTFS coverage).")
    return merged


# -- Main ----------------------------------------------------------------------

def main():
    stations = pd.read_csv(PROCESSED / "stations.csv", low_memory=False)
    print(f"Loaded {len(stations)} stations from stations.csv")

    # Reuse build_gtfs_features' GTFS loader and stop->code matcher
    rail_routes, trips_filt, st_filt, stops = load_gtfs()
    feat_df = compute_per_stop_features(rail_routes, trips_filt, st_filt, stops)
    feat_df = match_stops_to_stations(feat_df, stations)

    # Build stop_id -> station code map
    stop_to_code = dict(
        zip(feat_df["stop_id"], feat_df["matched_code"])
    )
    stop_to_code = {k: v for k, v in stop_to_code.items() if pd.notna(v)}

    print(f"\nBuilding directed edge list from {len(st_filt)} stop_times rows ...")
    edge_df = build_edge_list(st_filt, trips_filt, stop_to_code)

    print("\nComputing node features ...")
    graph_feats = compute_node_features(edge_df, stations)

    merged = join_to_stations(graph_feats, stations)

    out_path = PROCESSED / "stations.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nUpdated stations.csv written to {out_path}")

    print("\nGraph feature summary:")
    print(merged[GRAPH_FEATURE_COLS].describe().round(2).to_string())


if __name__ == "__main__":
    main()
