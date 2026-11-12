import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from features import CONGESTION_ORDER,DATASET_VERSION,PROCESSED_DIRECTORY
from ingest import TIMEZONE


PREDICTION_FILE = PROCESSED_DIRECTORY/f"predictions_{DATASET_VERSION}.csv"
SEVERITY_LEVELS = list(CONGESTION_ORDER)
SEVERITY_COLOURS = {"Free-flow": "#2e9e5b", "Moderate": "#f2c230", "Heavy": "#f08a24", "Severe": "#d0312d"}
UNKNOWN_COLOUR = "#9aa0a6"


@st.cache_data

def load_predictions(modified: float) -> pd.DataFrame :
    df = pd.read_csv(PREDICTION_FILE)
    for column in ["timestamp","forecast_for"] :
        df[column] = pd.to_datetime(df[column],utc=True).dt.tz_convert(TIMEZONE)
    return df.sort_values(["forecast_for","road_id"])

def pick_window(predictions:pd.DataFrame) -> pd.Timestamp :
    windows = predictions["forecast_for"].drop_duplicates().sort_values()
    days = windows.dt.date.unique()
    day = st.sidebar.selectbox("Date", days, index=len(days) - 1)
    times = windows[windows.dt.date == day]
    return st.sidebar.selectbox("Forecast time", times, index=len(times) - 1, format_func=lambda time: f"{time:%H:%M}")

def severity_colour(level: str) -> str:
    return SEVERITY_COLOURS.get(level, UNKNOWN_COLOUR)

def text_colour(level: str) -> str:
    return "#1f1f1f" if level == "Moderate" else "white"

def tile(label: str, value, level: str) -> str:
    return (
        f"<div style='background:{severity_colour(level)};color:{text_colour(level)};padding:12px;border-radius:6px;text-align:center'>"
        f"<div style='font-size:14px'>{label}</div><div style='font-size:32px;font-weight:600'>{value}</div></div>"
    )

def corridor_summary(rows: pd.DataFrame) -> None:
    counts = rows["predicted_congestion_level"].value_counts()
    columns = st.columns(len(SEVERITY_LEVELS))
    for column, level in zip(columns, SEVERITY_LEVELS):
        column.markdown(tile(level, counts.get(level, 0), level), unsafe_allow_html=True)

def colour_level(level: str) -> str:
    return f"background-color:{severity_colour(level)};color:{text_colour(level)}"

def segment_table(rows: pd.DataFrame) -> None:
    low = rows["predicted_volume_lower"].round().astype(int).astype(str)
    high = rows["predicted_volume_upper"].round().astype(int).astype(str)
    table = pd.DataFrame({
        "Road": rows["road_id"],
        "Congestion": rows["predicted_congestion_level"],
        "Confidence": rows["congestion_confidence"],
        "Volume": rows["predicted_volume"],
        "Volume range (90%)": low + " to " + high,
        "Travel time (min)": rows["predicted_travel_time"],
        "Accident risk": rows["accident_probability"],
    })
    styled = table.style.map(colour_level, subset=["Congestion"]).format(
        {"Confidence": "{:.0%}", "Volume": "{:.0f}", "Travel time (min)": "{:.1f}", "Accident risk": "{:.1%}"}
    )
    st.dataframe(styled, hide_index=True, width="stretch")
    st.caption("Accident risk is the chance of an incident on that road in this window. A typical window is about 1%, so 2% means roughly double the usual risk.")

def pick_road(rows: pd.DataFrame) -> pd.Series:
    road = st.sidebar.selectbox("Road", rows["road_id"])
    return rows[rows["road_id"] == road].iloc[0]

def road_detail(road: pd.Series) -> None:
    st.subheader(f"Road {road['road_id']}")
    level = road["predicted_congestion_level"]
    congestion, volume, travel, risk = st.columns(4)

    congestion.markdown(tile("Congestion", level, level), unsafe_allow_html=True)
    congestion.caption(f"Confidence {road['congestion_confidence']:.0%}")
    volume.metric("Volume (vehicles)", f"{road['predicted_volume']:.0f}")
    volume.caption(f"90% range: {road['predicted_volume_lower']:.0f} to {road['predicted_volume_upper']:.0f}")
    travel.metric("Travel time (min)", f"{road['predicted_travel_time']:.1f}")
    travel.caption(f"90% range: {road['predicted_travel_time_lower']:.1f} to {road['predicted_travel_time_upper']:.1f}")
    risk.metric("Accident risk", f"{road['accident_probability']:.1%}")


def main() -> None:
    st.set_page_config(page_title="FlowCast",layout="wide")
    st.title("FlowCast")

    Predictions = load_predictions(PREDICTION_FILE.stat().st_mtime)
    
    window = pick_window(Predictions)

    rows = Predictions[Predictions["forecast_for"]==window]
    st.write(f"forecast_for {window:%A %d %B %Y , %H : %M}")

    corridor_summary(rows)
    segment_table(rows)
    road_detail(pick_road(rows))



if __name__ == "__main__" :
    main()