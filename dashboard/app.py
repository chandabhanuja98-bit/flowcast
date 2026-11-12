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

def load_predictions() -> pd.DataFrame :
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

def corridor_summary(rows: pd.DataFrame) -> None:
    counts = rows["predicted_congestion_level"].value_counts()
    columns = st.columns(len(SEVERITY_LEVELS))
    for column, level in zip(columns, SEVERITY_LEVELS):
        text = "#1f1f1f" if level == "Moderate" else "white"
        column.markdown(
            f"<div style='background:{severity_colour(level)};color:{text};padding:12px;border-radius:6px;text-align:center'>"
            f"<div style='font-size:14px'>{level}</div><div style='font-size:32px;font-weight:600'>{counts.get(level, 0)}</div></div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="FlowCast",layout="wide")
    st.title("FlowCast")

    Predictions = load_predictions()
    
    window = pick_window(Predictions)

    rows = Predictions[Predictions["forecast_for"]==window]
    st.write(f"forecast_for {window:%A %d %B %Y , %H : %M}")

    corridor_summary(rows)
    st.dataframe(rows)



if __name__ == "__main__" :
    main()