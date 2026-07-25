"""
trends.py

Phase 4: aggregates logged predictions (from the database built in Phase 3)
into a time-series trend chart — sightings per species per week, smoothed
with a rolling average.

This is deliberately simple: pandas groupby + rolling mean, not a
forecasting model. It answers "is sighting frequency going up or down
over time," not "how many animals exist" (camera sighting frequency is a
proxy for activity/detection, not a population census).

Run from project root:
    python src/analysis/trends.py
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "api"))
from database import SessionLocal, Prediction

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Rolling average window, in weeks. 4 weeks smooths out week-to-week noise
# while still showing month-scale trend direction.
ROLLING_WINDOW_WEEKS = 4


def load_predictions_df():
    """
    Loads all predictions from the database into a pandas DataFrame.
    Excludes "null island" records (latitude=0, longitude=0) — these are
    leftover test calls made through /docs without real location data,
    not genuine sightings, and would otherwise show up as noise.
    """
    db = SessionLocal()
    records = db.query(Prediction).all()
    db.close()

    rows = [{
        "species": r.species,
        "confidence": r.confidence,
        "needs_review": r.needs_review,
        "latitude": r.latitude,
        "longitude": r.longitude,
        "created_at": r.created_at,
    } for r in records]

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])

    is_null_island = (df["latitude"] == 0) & (df["longitude"] == 0)
    df = df[~is_null_island]

    return df


def compute_weekly_counts(df):
    """
    Groups sightings into weekly buckets per species. Returns a table with
    one row per week, one column per species, values = sighting counts.
    Missing weeks (no sightings at all) get filled with 0 rather than
    being skipped, so the time axis stays continuous.
    """
    df = df.set_index("created_at")
    weekly = (
        df.groupby("species")
        .resample("W")
        .size()
        .unstack(level="species", fill_value=0)
    )
    return weekly


def plot_trends(weekly_counts, output_path):
    """
    Plots raw weekly counts (faint) and rolling-average smoothed trend
    (bold) per species, so both the noisy real data and the underlying
    trend direction are visible on the same chart.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for species in weekly_counts.columns:
        raw = weekly_counts[species]
        smoothed = raw.rolling(window=ROLLING_WINDOW_WEEKS, min_periods=1).mean()

        line, = ax.plot(smoothed.index, smoothed.values, linewidth=2, label=species)
        ax.plot(raw.index, raw.values, linewidth=0.5, alpha=0.3, color=line.get_color())

    ax.set_xlabel("Week")
    ax.set_ylabel("Sightings")
    ax.set_title(f"WildSense — Sightings per Week by Species "
                 f"({ROLLING_WINDOW_WEEKS}-week rolling average)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved trend chart to {output_path}")


def main():
    print("Loading predictions from database...")
    df = load_predictions_df()
    print(f"Loaded {len(df)} sightings (after excluding null-island test records)")

    if df.empty:
        print("No data to analyze. Make sure you've run seed_database.py first.")
        return

    print("\nSightings per species (all-time):")
    print(df["species"].value_counts().to_string())

    print(f"\nDate range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")

    weekly_counts = compute_weekly_counts(df)
    plot_trends(weekly_counts, RESULTS_DIR / "trend_chart.png")


if __name__ == "__main__":
    main()