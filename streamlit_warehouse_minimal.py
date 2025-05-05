"""
Warehouse Location Optimizer – Streamlit (no external ML deps)
=============================================================
Upload a CSV / Excel file with **Latitude**, **Longitude**, **Sales** columns (camel‑case like the
helper template). Choose how many warehouses you want, and the app assigns each
store to the nearest centroid found via a lightweight weighted K‑Means written
from scratch – so **no scikit‑learn** is required. All visualisations are built
with Altair for a zero‑compile install on Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import base64
from typing import Tuple

# ──────────────────────────────  Page setup  ──────────────────────────────
st.set_page_config(page_title="Warehouse Optimizer", page_icon="🏭", layout="wide")
st.title("Warehouse Location Optimizer")
st.markdown("""
Upload your store list, pick the number of warehouses, and instantly see the
optimal locations along with distance & sales metrics. 100‑store synth sample
included for quick testing.
""")

# ─────────────────────  Helper functions & core logic  ────────────────────

def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorised haversine (returns km). Accepts NumPy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def generate_sample_data(n: int = 100) -> pd.DataFrame:
    """Return a reproducible 100‑store dataset across the contiguous U.S."""
    rng = np.random.default_rng(42)
    lats = rng.uniform(25, 49, n)
    lons = rng.uniform(-124, -66, n)
    sales = rng.integers(10_000, 1_000_000, n)
    return pd.DataFrame({"Latitude": lats, "Longitude": lons, "Sales": sales})


def custom_kmeans(X: np.ndarray, k: int, weights: np.ndarray | None = None, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Simple weighted K‑Means (Euclidean) – returns (centroids, labels)."""
    n = X.shape[0]
    if weights is not None:
        # Sample initial centroids proportional to sales so big stores influence seeding
        weights_p = weights / weights.sum()
        centroids = X[np.random.choice(n, k, replace=False, p=weights_p)]
    else:
        centroids = X[np.random.choice(n, k, replace=False)]

    labels = np.full(n, -1)
    for _ in range(max_iter):
        # Assign step
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        # Update step
        for i in range(k):
            mask = labels == i
            if mask.any():
                if weights is None:
                    centroids[i] = X[mask].mean(axis=0)
                else:
                    w = weights[mask]
                    centroids[i] = (X[mask] * w[:, None]).sum(axis=0) / w.sum()
    return centroids, labels


def optimise_network(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster stores, compute distances & per‑warehouse metrics."""
    X = df[["Latitude", "Longitude"]].to_numpy()
    sales = df["Sales"].to_numpy()
    cents, labels = custom_kmeans(X, k, weights=sales)
    df = df.copy()
    df["Warehouse"] = labels
    # Distances in km to assigned centroid
    cents_lat = cents[labels, 0]
    cents_lon = cents[labels, 1]
    df["Distance_km"] = haversine_distance(df["Latitude"].values, df["Longitude"].values, cents_lat, cents_lon)
    # Centroids DF
    wh = pd.DataFrame(cents, columns=["Latitude", "Longitude"])
    wh["Warehouse"] = wh.index
    # Metrics
    metrics = (
        df.groupby("Warehouse")
        .agg(Stores=("Warehouse", "size"), Sales=("Sales", "sum"), Avg_Dist_km=("Distance_km", "mean"))
        .reset_index()
        .merge(wh, on="Warehouse")
    )
    return df, metrics


def df_download_link(df: pd.DataFrame, filename: str, text: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

# ───────────────────────────────  Sidebar  ────────────────────────────────
sidebar = st.sidebar
sidebar.header("Configuration")
use_sample = sidebar.checkbox("Use sample 100‑store dataset", value=True)

if use_sample:
    stores_df = generate_sample_data()
else:
    uploaded = sidebar.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx"])
    if uploaded is None:
        sidebar.info("Awaiting file...")
        st.stop()
    if uploaded.name.endswith(".csv"):
        stores_df = pd.read_csv(uploaded)
    else:
        stores_df = pd.read_excel(uploaded)

required = {"Latitude", "Longitude", "Sales"}
if not required.issubset(stores_df.columns):
    st.error(f"File must contain columns: {', '.join(required)}")
    st.stop()

k = sidebar.slider("Warehouses to optimise", 1, 10, 3)

# ─────────────────────────────  Main results  ─────────────────────────────
stores_df, metrics_df = optimise_network(stores_df, k)

# Tabs for UX
map_tab, metrics_tab, download_tab = st.tabs(["🗺 Map", "📊 Metrics", "⬇️ Downloads"])

with map_tab:
    st.subheader("Store & Warehouse Locations (Albers USA)")
    # Stores layer
    store_layer = (
        alt.Chart(stores_df)
        .mark_circle()
        .encode(
            longitude="Longitude:Q",
            latitude="Latitude:Q",
            size=alt.Size("Sales:Q", scale=alt.Scale(range=[30, 600]), title="Sales"),
            color=alt.Color("Warehouse:N", legend=None),
            tooltip=["Sales", "Distance_km"]
        )
    )
    # Warehouse layer – white‑border squares
    wh_layer = (
        alt.Chart(metrics_df)
        .mark_square(size=200, stroke="white", strokeWidth=2)
        .encode(
            longitude="Longitude:Q",
            latitude="Latitude:Q",
            color=alt.Color("Warehouse:N", legend=None),
            tooltip=["Warehouse", "Stores", "Avg_Dist_km"]
        )
    )
    chart = (store_layer + wh_layer).properties(width=900, height=550).project(type="albersUsa")
    st.altair_chart(chart, use_container_width=True)

with metrics_tab:
    st.subheader("Key metrics per warehouse")
    pretty = metrics_df.copy()
    pretty["Sales"] = pretty["Sales"].apply(lambda x: f"${x:,.0f}")
    pretty["Avg_Dist_km"] = pretty["Avg_Dist_km"].round(2)
    st.dataframe(pretty, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stores per warehouse**")
        st.altair_chart(
            alt.Chart(metrics_df).mark_bar().encode(x="Warehouse:N", y="Stores:Q", color="Warehouse:N"),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Total sales per warehouse**")
        st.altair_chart(
            alt.Chart(metrics_df).mark_bar().encode(x="Warehouse:N", y="Sales:Q", color="Warehouse:N"),
            use_container_width=True,
        )

with download_tab:
    st.subheader("Download CSVs")
    st.markdown(df_download_link(stores_df, "store_assignments.csv", "Store assignments"), unsafe_allow_html=True)
    st.markdown(df_download_link(metrics_df, "warehouse_metrics.csv", "Warehouse metrics"), unsafe_allow_html=True)

# ───────────────────────────  Footer tip  ────────────────────────────────
st.caption("No external ML libraries – deployment‑friendly on Streamlit Cloud ✨")
