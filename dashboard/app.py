import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from features import CONGESTION_ORDER,DATASET_VERSION,PROCESSED_DIRECTORY, INTERIM_FILE
from ingest import TIMEZONE
import plotly.graph_objects as go


PREDICTION_FILE = PROCESSED_DIRECTORY/f"predictions_{DATASET_VERSION}.csv"
SEVERITY_LEVELS = list(CONGESTION_ORDER)
SEVERITY_COLOURS = {"Free-flow": "#2e9e5b", "Moderate": "#f2c230", "Heavy": "#f08a24", "Severe": "#d0312d"}
UNKNOWN_COLOUR = "#9aa0a6"
HISTORY_COLUMNS = ["road_id","road_name","timestamp","traffic_volume", "avg_speed","travel_time","congestion_level"]
BAR_COLOUR = "#3b6ea8"
LONG_RANGE_DAYS = 14

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

@st.cache_data

def load_history(modified: float) -> pd.DataFrame:
    """Load the full cleaned history (actual readings, not predictions)."""
    df = pd.read_csv(INTERIM_FILE, usecols=HISTORY_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    return df.sort_values(["road_id", "timestamp"])

def pick_dates(history: pd.DataFrame, key: str) -> pd.DataFrame:
    """Let the user choose a date range and return only those rows."""
    first, last = history["timestamp"].min().date(), history["timestamp"].max().date()
    chosen = st.date_input("Date range", (last - pd.Timedelta(days=6), last), min_value=first, max_value=last, key=key)
    if len(chosen) != 2:
        st.info("Pick an end date to continue.")
        st.stop()
    start, end = chosen
    days = history["timestamp"].dt.date
    return history[(days >= start) & (days <= end)]

def line_chart(x, y, title: str, unit: str) -> go.Figure:
    figure = go.Figure(go.Scatter(x=x, y=y, mode="lines", line={"color": BAR_COLOUR}))
    figure.update_layout(title=title, yaxis_title=unit, height=300, margin={"t": 40, "b": 20})
    return figure

def trends_view(history: pd.DataFrame) -> None:
    """Volume and speed over time for one road, plus its daily peak-hour pattern."""
    road = st.selectbox("Road", history["road_id"].unique(), key="trend_road")
    rows = pick_dates(history, "trend_dates")
    rows = rows[rows["road_id"] == road].set_index("timestamp")
    if rows.empty:
        st.warning("No data for this selection.")
        return

    span = (rows.index.max() - rows.index.min()).days
    series = rows[["traffic_volume", "avg_speed"]]
    if span > LONG_RANGE_DAYS:
        series = series.resample("1h").mean()
        st.caption("Range is longer than two weeks, so points are hourly averages.")
    st.plotly_chart(line_chart(series.index, series["traffic_volume"], "Traffic volume over time", "vehicles per 30 min"), width="stretch")
    st.plotly_chart(line_chart(series.index, series["avg_speed"], "Average speed over time", "km/h"), width="stretch")

    slot = rows.index.hour + rows.index.minute / 60
    weekend = rows.index.dayofweek >= 5
    figure = go.Figure()
    for label, mask, colour in [("Weekday", ~weekend, BAR_COLOUR), ("Weekend", weekend, "#8a8f98")]:
        if mask.any():
            profile = rows.loc[mask, "traffic_volume"].groupby(slot[mask]).mean()
            figure.add_trace(go.Scatter(x=profile.index, y=profile.values, mode="lines", name=label, line={"color": colour}))
    figure.update_layout(title="Typical day: average volume by time of day (peak-hour pattern)", xaxis_title="Hour of day", yaxis_title="vehicles per 30 min", height=320, margin={"t": 40, "b": 20})
    st.plotly_chart(figure, width="stretch")

def severity_colourscale() -> list:
    """Four flat colour blocks, one per congestion level, so the heatmap has no blended shades."""
    scale = []
    for index, level in enumerate(SEVERITY_LEVELS):
        scale += [[index / len(SEVERITY_LEVELS), SEVERITY_COLOURS[level]], [(index + 1) / len(SEVERITY_LEVELS), SEVERITY_COLOURS[level]]]
    return scale

def heatmap_view(history: pd.DataFrame) -> None:
    """Grid of road x 30-minute window for one day, coloured by congestion level."""
    days = history["timestamp"].dt.date.unique()
    day = st.selectbox("Date", days, index=len(days) - 1, key="heatmap_day")
    rows = history[history["timestamp"].dt.date == day].copy()
    rows["slot"] = rows["timestamp"].dt.strftime("%H:%M")
    rows["code"] = rows["congestion_level"].map(CONGESTION_ORDER)

    codes = rows.pivot(index="road_id", columns="slot", values="code")
    levels = rows.pivot(index="road_id", columns="slot", values="congestion_level")
    figure = go.Figure(go.Heatmap(
        z=codes.values, x=codes.columns, y=codes.index,
        zmin=0, zmax=len(SEVERITY_LEVELS) - 1, colorscale=severity_colourscale(), showscale=False,
        customdata=levels.values, hovertemplate="Road %{y}<br>%{x}<br>%{customdata}<extra></extra>",
    ))
    figure.update_layout(height=650, xaxis_title="Time of day", yaxis_title="Road", yaxis={"autorange": "reversed", "type": "category"}, margin={"t": 20})
    st.plotly_chart(figure, width="stretch")
    legend = st.columns(len(SEVERITY_LEVELS))
    for column, level in zip(legend, SEVERITY_LEVELS):
        column.markdown(tile(level, "", level), unsafe_allow_html=True)

def road_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """One line per road: average volume, average speed and travel-time reliability."""
    grouped = rows.groupby("road_id")
    return pd.DataFrame({
        "Average volume": grouped["traffic_volume"].mean(),
        "Average speed (km/h)": grouped["avg_speed"].mean(),
        "Travel-time variability (%)": grouped["travel_time"].std() / grouped["travel_time"].mean() * 100,
    })

def bar_chart(values: pd.Series, title: str) -> go.Figure:
    values = values.sort_values(ascending=False)
    figure = go.Figure(go.Bar(x=values.index, y=values.values, marker_color=BAR_COLOUR))
    figure.update_layout(title=title, height=300, margin={"t": 40, "b": 20}, xaxis={"type": "category"})
    return figure

def comparison_view(history: pd.DataFrame) -> None:
    """Side-by-side comparison of roads on volume, speed and reliability."""
    rows = pick_dates(history, "compare_dates")
    roads = st.multiselect("Roads to compare", sorted(rows["road_id"].unique()), default=sorted(rows["road_id"].unique()))
    summary = road_summary(rows[rows["road_id"].isin(roads)])
    if summary.empty:
        st.warning("Pick at least one road.")
        return
    for column in summary.columns:
        st.plotly_chart(bar_chart(summary[column], column), width="stretch")
    st.caption("Travel-time variability is the standard deviation of travel time divided by its average. A lower value means a more reliable road.")

def live_view() -> None:
    """The existing live-prediction view."""
    predictions = load_predictions(PREDICTION_FILE.stat().st_mtime)
    window = pick_window(predictions)
    rows = predictions[predictions["forecast_for"] == window]
    st.write(f"forecast_for {window:%A %d %B %Y , %H : %M}")
    corridor_summary(rows)
    segment_table(rows)
    road_detail(pick_road(rows))

    
def main() -> None:
    st.set_page_config(page_title="FlowCast", layout="wide")
    st.title("FlowCast")
    history = load_history(INTERIM_FILE.stat().st_mtime)

    live, trends, heatmap, comparison = st.tabs(["Live prediction", "Historical trends", "Congestion heatmap", "Road comparison"])
    with live:
        live_view()
    with trends:
        trends_view(history)
    with heatmap:
        heatmap_view(history)
    with comparison:
        comparison_view(history)




if __name__ == "__main__" :
    main()