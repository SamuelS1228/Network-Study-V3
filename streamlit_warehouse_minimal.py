"""
Warehouse Location Optimizer (Streamlit app)
Upload a CSV/XLSX file with columns: lat, lon, sales.
Select the number of warehouses to optimize for and visualize the result.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import altair as alt
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Warehouse Location Optimizer", layout="wide")

# -----------------------------------------------------------------------------
# Sidebar – controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣  Upload Your Store Data")
    uploaded_file = st.file_uploader(
        "CSV or Excel with columns: lat, lon, sales", type=["csv", "xlsx"]
    )
    use_sample = st.checkbox("Use built‑in sample dataset (100 stores)")

    st.header("2️⃣  Choose Number of Warehouses")
    n_clusters = st.slider(
        "Warehouses to solve for", min_value=1, max_value=10, value=3
    )

    st.markdown("---")
    st.caption(
        "Tip: pinning dependency versions in **requirements.txt** avoids Streamlit Cloud build issues."
    )

# -----------------------------------------------------------------------------
# Data ingestion & validation
# -----------------------------------------------------------------------------

def load_data(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

required_cols = {"lat", "lon", "sales"}

if use_sample:
    # Generate a reproducible synthetic dataset (100 stores across contiguous U.S.)
    rng = np.random.default_rng(42)
    lats = rng.uniform(25, 49, 100)
    lons = rng.uniform(-124, -66, 100)
    sales = rng.integers(10_000, 1_000_000, 100)
    stores_df = pd.DataFrame({"lat": lats, "lon": lons, "sales": sales})
elif uploaded_file is not None:
    stores_df = load_data(uploaded_file)
else:
    st.info("⬅️ Upload a dataset or select the sample option to begin.")
    st.stop()

if not required_cols.issubset(stores_df.columns):
    st.error(f"Data file must contain columns: {', '.join(required_cols)}")
    st.stop()

# -----------------------------------------------------------------------------
# Weighted K‑Means clustering (sales as weights)
# -----------------------------------------------------------------------------
coords = stores_df[["lat", "lon"]].to_numpy()
weights = stores_df["sales"].to_numpy()

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
kmeans.fit(coords, sample_weight=weights)

stores_df["warehouse_id"] = kmeans.predict(coords)
centroids = kmeans.cluster_centers_
warehouses_df = (
    pd.DataFrame(centroids, columns=["lat", "lon"]).assign(warehouse_id=lambda d: d.index)
)

# -----------------------------------------------------------------------------
# Distance calculation (vectorized Haversine) – result in miles
# -----------------------------------------------------------------------------
R = 3958.8  # Earth radius in miles
lat1 = np.radians(stores_df["lat"].values)
lon1 = np.radians(stores_df["lon"].values)
lat2 = np.radians(centroids[stores_df["warehouse_id"], 0])
lon2 = np.radians(centroids[stores_df["warehouse_id"], 1])

dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
stores_df["distance_mi"] = R * 2 * np.arcsin(np.sqrt(a))

# -----------------------------------------------------------------------------
# Summary metrics per warehouse
# -----------------------------------------------------------------------------
summary_df = (
    stores_df.groupby("warehouse_id").agg(
        num_stores=("warehouse_id", "size"),
        total_sales=("sales", "sum"),
        avg_distance_mi=("distance_mi", "mean"),
    )
    .reset_index()
    .merge(warehouses_df, on="warehouse_id")
)

# -----------------------------------------------------------------------------
# Map visualisation (Folium rendered via streamlit‑folium)
# -----------------------------------------------------------------------------
map_center = [stores_df["lat"].mean(), stores_df["lon"].mean()]
folium_map = folium.Map(location=map_center, zoom_start=5, control_scale=True)

# Stores – tiny circle markers
for _, row in stores_df.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=3,
        weight=0,
        fill=True,
        fill_opacity=0.6,
        popup=f"Sales: {row.sales:,.0f} USD",
    ).add_to(folium_map)

# Warehouses – pin icons
for _, row in warehouses_df.iterrows():
    folium.Marker(
        location=[row.lat, row.lon],
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
        popup=f"Warehouse {row.warehouse_id}",
    ).add_to(folium_map)

st.subheader("Network Map")
st_folium(folium_map, height=600, use_container_width=True)

# -----------------------------------------------------------------------------
# Charts & tables
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Stores per Warehouse")
    bar = (
        alt.Chart(summary_df)
        .mark_bar()
        .encode(x="warehouse_id:N", y="num_stores:Q", tooltip=["num_stores", "total_sales"])
    )
    st.altair_chart(bar, use_container_width=True)

with col2:
    st.subheader("Key Metrics")
    st.dataframe(
        summary_df.rename(
            columns={
                "warehouse_id": "Warehouse",
                "num_stores": "Stores",
                "total_sales": "Total Sales",
                "avg_distance_mi": "Avg Dist (mi)",
            }
        ),
        hide_index=True,
    )

# -----------------------------------------------------------------------------
# Download buttons
# -----------------------------------------------------------------------------
@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode("utf‑8")

st.download_button(
    "Download Store Assignments (CSV)", to_csv(stores_df), "store_assignments.csv", "text/csv"
)

st.download_button(
    "Download Warehouse Summary (CSV)", to_csv(summary_df), "warehouse_summary.csv", "text/csv"
)
