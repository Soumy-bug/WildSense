import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px

sys.path.append(str(Path(__file__).parent.parent / "api"))
from database import SessionLocal, Prediction

ROLLING_WINDOW_WEEKS = 4

st.set_page_config(page_title="WildSense Dashboard", layout="wide")


SNAPSHOT_PATH = Path("data/processed/predictions_snapshot.csv")


@st.cache_data(ttl=60)  
def load_predictions():
    """
    Loads predictions from the live database if it has data. Falls back to
    a committed CSV snapshot otherwise — this matters specifically for the
    deployed version of this dashboard (Streamlit Community Cloud), which
    has no access to the local wildsense.db that Phase 3's API writes to.
    The snapshot is a point-in-time export (see export_snapshot.py) of the
    same seeded historical data used throughout local development.
    """
    df = pd.DataFrame()

    try:
        db = SessionLocal()
        records = db.query(Prediction).all()
        db.close()

        if records:
            rows = [{
                "species": r.species,
                "confidence": r.confidence,
                "needs_review": r.needs_review,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "created_at": r.created_at,
            } for r in records]
            df = pd.DataFrame(rows)
    except Exception:
        pass 

    if df.empty and SNAPSHOT_PATH.exists():
        df = pd.read_csv(SNAPSHOT_PATH)

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
    is_null_island = (df["latitude"] == 0) & (df["longitude"] == 0)
    df = df[~is_null_island]
    return df


def build_map(df):
    """
    Builds a Folium map with two layers: a density heatmap (shows overall
    activity concentration at a glance) and individual clustered markers
    (lets you click into specific sightings). Both respond to whatever
    species filter is currently active.
    """
    if df.empty:
        center = [31.9, -110.0]
    else:
        center = [df["latitude"].mean(), df["longitude"].mean()]

    fmap = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    if not df.empty:
        heat_points = df[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_points, radius=15, blur=20).add_to(
            folium.FeatureGroup(name="Sighting density").add_to(fmap)
        )

        cluster = MarkerCluster(name="Individual sightings").add_to(fmap)
        for _, row in df.iterrows():
            popup_text = (
                f"<b>{row['species']}</b><br>"
                f"Confidence: {row['confidence']:.2f}<br>"
                f"Date: {row['created_at'].strftime('%Y-%m-%d')}"
            )
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color="green" if not row["needs_review"] else "orange"),
            ).add_to(cluster)

    folium.LayerControl().add_to(fmap)
    return fmap


def build_trend_chart(df):
    """
    Weekly sighting counts per species, smoothed with a rolling average —
    same logic as Phase 4's trends.py, but interactive (Plotly) and
    responsive to the current species filter.
    """
    if df.empty:
        return None

    weekly = (
        df.set_index("created_at")
        .groupby("species")
        .resample("W")
        .size()
        .reset_index(name="sightings")
    )

    weekly["smoothed"] = (
        weekly.groupby("species")["sightings"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_WEEKS, min_periods=1).mean())
    )

    fig = px.line(
        weekly, x="created_at", y="smoothed", color="species",
        labels={"created_at": "Week", "smoothed": "Sightings (rolling avg)"},
        title=f"Sightings per week ({ROLLING_WINDOW_WEEKS}-week rolling average)",
    )
    return fig


def build_time_of_day_chart(df):
    """
    Shows sightings grouped by hour of day, per species — reveals activity
    patterns like nocturnal vs. diurnal behavior. This is genuine ecological
    signal that's essentially free, since every prediction already has a
    real timestamp — no new data collection needed, just a different way
    of grouping what we already have.
    """
    if df.empty:
        return None

    df = df.copy()
    df["hour"] = df["created_at"].dt.hour

    hourly = df.groupby(["hour", "species"]).size().reset_index(name="sightings")

    fig = px.bar(
        hourly, x="hour", y="sightings", color="species", barmode="group",
        labels={"hour": "Hour of day (24h)", "sightings": "Sightings"},
        title="Activity by time of day",
    )
    fig.update_xaxes(dtick=2, range=[-0.5, 23.5])
    return fig


def build_confidence_histogram(df):
    """
    Shows the distribution of prediction confidence scores across all
    sightings — a diagnostic companion to the triage feature. A healthy
    distribution should show most predictions clustered high (>0.7), with
    a visible tail below the 0.5 triage threshold representing the
    "needs review" cases.
    """
    if df.empty:
        return None

    fig = px.histogram(
        df, x="confidence", color="needs_review", nbins=20,
        labels={"confidence": "Prediction confidence", "needs_review": "Needs review"},
        title="Distribution of prediction confidence",
        color_discrete_map={True: "orange", False: "seagreen"},
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                   annotation_text="triage threshold")
    return fig


def main():
    st.title("🦨 WildSense — Species Monitoring Dashboard")
    st.caption(
        "Camera trap sightings identified automatically by a fine-tuned "
        "classifier. GPS coordinates are simulated for demonstration — "
        "see README for details."
    )

    df = load_predictions()

    if df.empty:
        st.warning("No sightings logged yet. Run seed_database.py or make some /predict calls first.")
        return

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total sightings", len(df))
    metric_col2.metric("Species tracked", df["species"].nunique())
    metric_col3.metric(
        "Date range",
        f"{df['created_at'].min().year}–{df['created_at'].max().year}",
    )
    metric_col4.metric("Avg. confidence", f"{df['confidence'].mean():.0%}")

    st.divider()

    all_species = sorted(df["species"].unique())
    selected_species = st.sidebar.multiselect(
        "Filter by species", options=all_species, default=all_species
    )
    show_review_only = st.sidebar.checkbox("Show only low-confidence (needs review)", value=False)

    min_date = df["created_at"].min().date()
    max_date = df["created_at"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )

    filtered = df[df["species"].isin(selected_species)]
    if show_review_only:
        filtered = filtered[filtered["needs_review"]]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["created_at"].dt.date >= start_date)
            & (filtered["created_at"].dt.date <= end_date)
        ]

    st.sidebar.markdown("---")
    st.sidebar.metric("Total sightings shown", len(filtered))
    st.sidebar.write(filtered["species"].value_counts())

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Sighting map")
        fmap = build_map(filtered)
        st_folium(fmap, width=700, height=500)

    with col2:
        st.subheader("Trend over time")
        fig = build_trend_chart(filtered)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for current filter.")

    st.subheader("Activity by time of day")
    time_fig = build_time_of_day_chart(filtered)
    if time_fig is not None:
        st.plotly_chart(time_fig, use_container_width=True)
    else:
        st.info("No data for current filter.")

    st.subheader("Prediction confidence distribution")
    conf_fig = build_confidence_histogram(filtered)
    if conf_fig is not None:
        st.plotly_chart(conf_fig, use_container_width=True)
    else:
        st.info("No data for current filter.")

    st.subheader("Raw sighting data")
    display_df = filtered.sort_values("created_at", ascending=False).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True)
    st.download_button(
        "Download filtered data as CSV",
        data=display_df.to_csv(index=False),
        file_name="wildsense_sightings.csv",
        mime="text/csv",
    )

    with st.expander("About this data & known limitations"):
        st.markdown(
            """
            **GPS coordinates are simulated.** The source dataset (CCT-20)
            provides camera location IDs, not real GPS. Each location was
            assigned a fixed, plausible coordinate for demonstration —
            these are not the cameras' real positions.

            **Confidence-based triage:** predictions below 0.5 confidence
            are flagged "needs review" (shown as orange markers) rather
            than logged as confirmed sightings — mirroring how a real
            deployment would route uncertain cases to a human reviewer
            instead of trusting every automated call.

            **Model limitations:** trained on 6 species with an overall
            validation macro F1 of 0.906. Bobcat and coyote are the most
            frequently confused pair, likely due to similar body shape and
            coloring in camera trap conditions. Rare species (e.g. skunk,
            ~200 training images) have less data to learn from, and their
            trend lines above are noisier and less reliable as a result.

            **Trend charts show sighting frequency, not population size.**
            These are a proxy for activity/detection rate, not a
            population census — actual population estimates require
            different methods (e.g. capture-recapture) that account for
            detection probability.
            """
        )


if __name__ == "__main__":
    main()