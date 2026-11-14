import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from features import CONGESTION_ORDER,DATASET_VERSION,PROCESSED_DIRECTORY, INTERIM_FILE
from ingest import PROJECT_ROOT, TIMEZONE
import insights
from evaluate import regression_metrics
from utils import load_model, model_path
import plotly.graph_objects as go


PREDICTION_FILE = PROCESSED_DIRECTORY/f"predictions_{DATASET_VERSION}.csv"
SEVERITY_LEVELS = list(CONGESTION_ORDER)
SEVERITY_COLOURS = {"Free-flow": "#2e9e5b", "Moderate": "#f2c230", "Heavy": "#f08a24", "Severe": "#d0312d"}
UNKNOWN_COLOUR = "#9aa0a6"
HISTORY_COLUMNS = ["road_id","road_name","timestamp","traffic_volume", "avg_speed","travel_time","congestion_level",
                   "weather_condition","rainfall","visibility","accident_count"]
WEATHER_ORDER = ["clear", "cloudy", "overcast", "rain", "fog"]
BAR_COLOUR = "#3b6ea8"
LONG_RANGE_DAYS = 14
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
SCOREBOARD_FILES = ["regression_scoreboard.csv", "classification_scoreboard.csv", "dl_scoreboard.csv"]
GRU_PREDICTION_FILE = PROCESSED_DIRECTORY / f"predictions_gru_{DATASET_VERSION}.csv"
DEEP_COLOUR = "#7a5aa6"
DEEP_MODELS = ("GRU", "LSTM")
ENSEMBLE_MODELS = ["Random Forest", "XGBoost"]
TARGET_LABELS = {
    "target_volume": "Traffic volume",
    "target_travel_time": "Travel time",
    "target_congestion": "Congestion level",
    "target_accident": "Accident risk",
}
PRIMARY_METRICS = {
    "target_volume": ("RMSE", False),
    "target_travel_time": ("RMSE", False),
    "target_congestion": ("macro_F1", True),
    "target_accident": ("ROC_AUC", True),
}
PRD_TARGET_LINES = {"target_congestion": 0.80, "target_accident": 0.75}
FORECAST_MEASURES = {"Volume": "volume", "Travel time": "travel_time"}
FORECAST_UNITS = {"volume": "vehicles per 30 min", "travel_time": "minutes"}

@st.cache_data

def load_predictions(modified: float) -> pd.DataFrame :
    df = pd.read_csv(PREDICTION_FILE)
    for column in ["timestamp","forecast_for"] :
        df[column] = pd.to_datetime(df[column],utc=True).dt.tz_convert(TIMEZONE)
    return df.sort_values(["forecast_for","road_id"])

def widget_key(prefix: str, name: str):
    return prefix + name if prefix else None

def pick_window(predictions:pd.DataFrame, date_box, time_box, prefix: str = "") -> pd.Timestamp :
    windows = predictions["forecast_for"].drop_duplicates().sort_values()
    days = windows.dt.date.unique()
    day = date_box.selectbox("Date", days, index=len(days) - 1, key=widget_key(prefix, "date"))
    times = windows[windows.dt.date == day]
    return time_box.selectbox("Forecast time", times, index=len(times) - 1, format_func=lambda time: f"{time:%H:%M}", key=widget_key(prefix, "time"))

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

def pick_road(rows: pd.DataFrame, road_box, prefix: str = "") -> pd.Series:
    road = road_box.selectbox("Road", rows["road_id"], key=widget_key(prefix, "road"))
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
    return insights.add_weather_columns(df.sort_values(["road_id", "timestamp"]))

def pick_dates(history: pd.DataFrame, key: str, column: str = "timestamp", default_days: int = 7) -> pd.DataFrame:
    """Let the user choose a date range and return only those rows."""
    first, last = history[column].min().date(), history[column].max().date()
    chosen = st.date_input("Date range", (max(first, last - pd.Timedelta(days=default_days - 1)), last), min_value=first, max_value=last, key=key)
    if len(chosen) != 2:
        st.info("Pick an end date to continue.")
        st.stop()
    start, end = chosen
    days = history[column].dt.date
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
    date_box, time_box, road_box = st.columns(3)
    window = pick_window(predictions, date_box, time_box)
    rows = predictions[predictions["forecast_for"] == window]
    road = pick_road(rows, road_box)
    st.write(f"forecast_for {window:%A %d %B %Y , %H : %M}")
    corridor_summary(rows)
    segment_table(rows)
    road_detail(road)

    
def scoreboard_files() -> list[Path]:
    """The scoreboard csv files that exist in reports/."""
    return [REPORTS_DIRECTORY / name for name in SCOREBOARD_FILES if (REPORTS_DIRECTORY / name).exists()]


@st.cache_data
def load_scoreboard(modified: float) -> pd.DataFrame:
    """All model scores (classical and deep) in one table."""
    return pd.concat([pd.read_csv(file) for file in scoreboard_files()], ignore_index=True)


def is_deep(model: str) -> bool:
    return model.split(" ")[0] in DEEP_MODELS


def criterion(column, label: str, value: str, target: str, met: bool, model: str) -> None:
    """One success-criterion tile: the best value reached, the PRD target and whether it was met."""
    column.metric(label, value)
    column.caption(f"Target {target}: {'Met' if met else 'Not met'} ({model})")


def success_criteria(test: pd.DataFrame) -> None:
    """The PRD success criteria, worked out from the test scores."""
    volume = test[test["target"] == "target_volume"]
    best_mape = volume.loc[volume["MAPE"].idxmin()]
    congestion = test[test["target"] == "target_congestion"]
    best_f1 = congestion.loc[congestion["macro_F1"].idxmax()]
    accident = test[test["target"] == "target_accident"].dropna(subset=["ROC_AUC"])
    best_auc = accident.loc[accident["ROC_AUC"].idxmax()]
    deep = volume[volume["model"].map(is_deep)]
    classical = volume[~volume["model"].map(is_deep)]

    st.subheader("PRD success criteria (test window)")
    tiles = st.columns(4)
    criterion(tiles[0], "Volume MAPE", f"{best_mape['MAPE']:.2f}%", "<= 12%", best_mape["MAPE"] <= 12, best_mape["model"])
    criterion(tiles[1], "Congestion macro-F1", f"{best_f1['macro_F1']:.3f}", ">= 0.80", best_f1["macro_F1"] >= 0.80, best_f1["model"])
    criterion(tiles[2], "Accident ROC-AUC", f"{best_auc['ROC_AUC']:.3f}", ">= 0.75", best_auc["ROC_AUC"] >= 0.75, best_auc["model"])
    if not deep.empty and not classical.empty:
        deep_best, classical_best = deep.loc[deep["RMSE"].idxmin()], classical.loc[classical["RMSE"].idxmin()]
        criterion(tiles[3], "Deep vs classical (volume RMSE)", f"{deep_best['RMSE']:.2f} vs {classical_best['RMSE']:.2f}",
                  "deep model lower", deep_best["RMSE"] < classical_best["RMSE"], f"{deep_best['model']} vs {classical_best['model']}")


def score_chart(part: pd.DataFrame, target: str) -> go.Figure:
    """Bar chart of the main metric for every model, with the PRD target line where there is one."""
    metric, higher_is_better = PRIMARY_METRICS[target]
    figure = go.Figure(go.Bar(
        x=part[metric], y=part["model"], orientation="h",
        marker_color=[DEEP_COLOUR if is_deep(model) else BAR_COLOUR for model in part["model"]],
    ))
    if target in PRD_TARGET_LINES:
        figure.add_vline(x=PRD_TARGET_LINES[target], line_dash="dash", annotation_text="PRD target")
    direction = "higher is better" if higher_is_better else "lower is better"
    figure.update_layout(title=f"{metric} on the test window ({direction}). Blue = classical, purple = deep",
                         height=max(300, 45 * len(part)), margin={"t": 40, "b": 20}, yaxis={"autorange": "reversed", "automargin": True})
    return figure


def loss_curves() -> None:
    """Training and validation loss of each deep model, one chart per model."""
    curves = {cell: REPORTS_DIRECTORY / f"dl_loss_curve_{cell}.csv" for cell in ["gru", "lstm"]}
    curves = {cell: file for cell, file in curves.items() if file.exists()}
    if not curves:
        return
    st.subheader("Deep model training curves")
    for column, (cell, file) in zip(st.columns(len(curves)), curves.items()):
        history = pd.read_csv(file)
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], mode="lines", name="Training loss", line={"color": BAR_COLOUR}))
        figure.add_trace(go.Scatter(x=history["epoch"], y=history["validation_loss"], mode="lines", name="Validation loss", line={"color": DEEP_COLOUR}))
        figure.update_layout(title=cell.upper(), xaxis_title="Epoch", yaxis_title="Loss", height=300, margin={"t": 40, "b": 20})
        column.plotly_chart(figure, width="stretch")
    st.caption("Training stops early when the validation loss stops improving, which guards against memorising the training data.")


def performance_view() -> None:
    """Test scores of every model, the PRD success criteria and the deep model training curves."""
    files = scoreboard_files()
    if not files:
        st.warning("No scoreboard files found in reports/. Train the models first.")
        return
    board = load_scoreboard(max(file.stat().st_mtime for file in files))
    test = board[board["split"] == "test"]
    success_criteria(test)

    st.subheader("Model comparison")
    target = st.selectbox("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, key="performance_target")
    metric, higher_is_better = PRIMARY_METRICS[target]
    part = test[test["target"] == target].drop(columns=["target", "split"]).dropna(axis=1, how="all")
    part = part.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    best = part.style.highlight_max(subset=[metric], color="#cfe3f7") if higher_is_better else part.style.highlight_min(subset=[metric], color="#cfe3f7")
    st.dataframe(best.format({column: "{:.3f}" for column in part.columns if column != "model"}), hide_index=True, width="stretch")
    st.plotly_chart(score_chart(part, target), width="stretch")
    loss_curves()


@st.cache_data
def load_importance(target: str, name: str, modified: float) -> pd.Series:
    """Feature importance of one saved tree ensemble, largest first."""
    bundle = load_model(name, target)
    return pd.Series(bundle["model"].feature_importances_, index=bundle["feature_columns"]).sort_values(ascending=False)


def importance_view() -> None:
    """Which input features each tree ensemble relies on most."""
    target = st.selectbox("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, key="importance_target")
    available = [name for name in ENSEMBLE_MODELS if model_path(name, target, "joblib").exists()]
    if not available:
        st.warning("No saved Random Forest or XGBoost model for this target.")
        return
    name = st.radio("Model", available, horizontal=True, key="importance_model")
    top = st.slider("Features to show", 5, 30, 15, key="importance_top")

    path = model_path(name, target, "joblib")
    with st.spinner("Loading model..."):
        importance = load_importance(target, name, path.stat().st_mtime)
    shown = importance.head(top)
    figure = go.Figure(go.Bar(x=shown.values, y=shown.index, orientation="h", marker_color=BAR_COLOUR))
    figure.update_layout(title=f"{name}: top {top} features for {TARGET_LABELS[target].lower()}",
                         height=max(300, 28 * top), margin={"t": 40, "b": 20}, yaxis={"autorange": "reversed", "automargin": True}, xaxis_title="Importance")
    st.plotly_chart(figure, width="stretch")
    st.caption(f"The top {top} of {len(importance)} features carry {shown.sum():.0%} of the total importance. "
               "A high value means the model leans on that feature a lot. It shows what the model uses, not what causes traffic.")


@st.cache_data
def load_forecasts(path: str, modified: float) -> pd.DataFrame:
    """Predictions together with the real values, from one predictions file."""
    df = pd.read_csv(path)
    for column in ["timestamp", "forecast_for"]:
        df[column] = pd.to_datetime(df[column], utc=True).dt.tz_convert(TIMEZONE)
    return df.sort_values(["road_id", "forecast_for"])


def forecast_chart(rows: pd.DataFrame, measure: str) -> go.Figure:
    """Actual and predicted values over time, with the 90% confidence band around the prediction."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=rows["forecast_for"], y=rows[f"predicted_{measure}_upper"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=rows["forecast_for"], y=rows[f"predicted_{measure}_lower"], mode="lines", line={"width": 0},
                                fill="tonexty", fillcolor="rgba(59,110,168,0.18)", name="90% confidence band"))
    figure.add_trace(go.Scatter(x=rows["forecast_for"], y=rows[f"actual_{measure}"], mode="lines", name="Actual", line={"color": "#444444"}))
    figure.add_trace(go.Scatter(x=rows["forecast_for"], y=rows[f"predicted_{measure}"], mode="lines", name="Predicted", line={"color": BAR_COLOUR}))
    figure.update_layout(yaxis_title=FORECAST_UNITS[measure], height=420, margin={"t": 20, "b": 20}, legend={"orientation": "h"})
    return figure


def forecast_view() -> None:
    """Predicted against actual for one road, with the confidence band and the error of what is shown."""
    sources = {"Classical winners": PREDICTION_FILE, "GRU": GRU_PREDICTION_FILE}
    sources = {label: path for label, path in sources.items() if path.exists()}
    if not sources:
        st.warning("No predictions file found in data/processed/. Run the prediction step first.")
        return
    model_box, measure_box, road_box = st.columns(3)
    source = model_box.selectbox("Model", list(sources), key="forecast_model")
    path = sources[source]
    data = load_forecasts(str(path), path.stat().st_mtime)
    measures = {label: key for label, key in FORECAST_MEASURES.items() if f"predicted_{key}" in data.columns}
    measure = measures[measure_box.selectbox("Measure", list(measures), key="forecast_measure")]
    road = road_box.selectbox("Road", sorted(data["road_id"].unique()), key="forecast_road")

    rows = pick_dates(data[data["road_id"] == road], "forecast_dates", "forecast_for", 2)
    st.plotly_chart(forecast_chart(rows, measure), width="stretch")

    scores = regression_metrics(rows[f"actual_{measure}"].to_numpy(dtype=float), rows[f"predicted_{measure}"].to_numpy(dtype=float))
    inside = (rows[f"actual_{measure}"] >= rows[f"predicted_{measure}_lower"]) & (rows[f"actual_{measure}"] <= rows[f"predicted_{measure}_upper"])
    for column, (label, value) in zip(st.columns(4), [("MAE", f"{scores['MAE']:.1f}"), ("RMSE", f"{scores['RMSE']:.1f}"), ("MAPE", f"{scores['MAPE']:.1f}%"), ("Actuals inside the band", f"{inside.mean():.0%}")]):
        column.metric(label, value)
    test_start = data.loc[data["split"] == "test", "forecast_for"].min()
    st.caption(f"These numbers are for the road and dates shown. Windows before {test_start:%d %b %Y} are validation data used to pick the models. "
               "From that date on they are the unseen test window. The band is built to hold 90% of actual values.")


def signed_bar(labels, values, title: str, unit: str) -> go.Figure:
    """Bars in the given order, with a zero line so drops below normal are easy to see."""
    figure = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=BAR_COLOUR))
    figure.add_hline(y=0, line_width=1)
    figure.update_layout(title=title, yaxis_title=unit, height=300, margin={"t": 40, "b": 20}, xaxis={"type": "category"})
    return figure


def effect_section(history: pd.DataFrame, column: str, heading: str, order: list[str] | None = None) -> None:
    """Volume, speed and incident charts plus the table for one way of grouping the weather."""
    table = insights.weather_effect(history, column)
    if order:
        table = table.reindex([item for item in order if item in table.index])
    st.subheader(heading)
    charts = [
        ("Volume vs typical (%)", "Traffic volume vs typical", "% vs typical"),
        ("Speed vs typical (%)", "Average speed vs typical", "% vs typical"),
        ("Incidents per 1,000 windows", "Incidents", "per 1,000 windows"),
    ]
    for box, (name, title, unit) in zip(st.columns(len(charts)), charts):
        box.plotly_chart(signed_bar(table.index.astype(str), table[name], title, unit), width="stretch")
    formats = {name: "{:.1f}" for name in table.columns} | {"Windows": "{:,.0f}"}
    st.dataframe(table.style.format(formats), width="stretch")


def weather_forecast_error(history: pd.DataFrame) -> None:
    """How accurate the volume forecast is in each weather condition."""
    if not PREDICTION_FILE.exists():
        return
    forecasts = load_forecasts(str(PREDICTION_FILE), PREDICTION_FILE.stat().st_mtime)
    table = insights.forecast_error_by_weather(forecasts, history)
    table = table.reindex([item for item in WEATHER_ORDER if item in table.index])
    st.subheader("Forecast error by weather (validation and test windows)")
    st.plotly_chart(signed_bar(table.index, table["Forecast error, MAPE (%)"], "Volume forecast error (MAPE) by weather condition", "MAPE (%)"), width="stretch")
    st.dataframe(table.style.format({column: "{:.1f}" for column in table.columns} | {"Windows": "{:,.0f}"}), width="stretch")
    st.caption("Fog and rain windows are rare, so look at the Windows column before trusting their error figures.")


def weather_view(history: pd.DataFrame) -> None:
    """How weather relates to observed traffic and to forecast accuracy."""
    st.write("Every reading is compared with what is normal for the same road, the same time of day and the same day type "
             "(weekday or weekend), so rush hour does not hide the effect of the weather.")
    effect_section(history, "weather_condition", "By weather condition", WEATHER_ORDER)
    effect_section(history, "rain_band", "By rainfall in the window")
    effect_section(history, "visibility_band", "By visibility")
    st.caption("The Heavy or Severe share drops in bad weather because fewer vehicles are on the road, not because traffic moves better.")
    weather_forecast_error(history)


def start_upload_run(pipeline, traffic, weather, calendar) -> None:
    """Run the pipeline on the uploaded files and show each stage as it happens."""
    files = {pipeline.TRAFFIC_FILE: traffic, pipeline.WEATHER_FILE: weather, pipeline.CALENDAR_FILE: calendar}
    uploads = {name: file.getvalue() if file else None for name, file in files.items()}
    names = {name: file.name for name, file in files.items() if file}
    with st.status("Running the pipeline...", expanded=True) as status:
        try:
            pipeline.run_upload(uploads, names, status.write)
        except pipeline.UploadError as error:
            status.update(label="The run stopped", state="error")
            st.error(str(error))
            return
        status.update(label="Prediction run finished", state="complete", expanded=False)


def upload_scores(predictions: pd.DataFrame, info: dict) -> None:
    """Check the forecasts against the real values that came with the uploaded file."""
    scores = regression_metrics(predictions["actual_volume"].to_numpy(dtype=float), predictions["predicted_volume"].to_numpy(dtype=float))
    accuracy = (predictions["predicted_congestion"] == predictions["actual_congestion"]).mean()
    for box, (label, value) in zip(st.columns(3), [("Volume MAPE", f"{scores['MAPE']:.1f}%"), ("Volume MAE", f"{scores['MAE']:.1f}"), ("Congestion level correct", f"{accuracy:.0%}")]):
        box.metric(label, value)
    trained_until = pd.Timestamp(info["models"]["target_volume"]["training_window_to"])
    if pd.Timestamp(info["forecast_from"]) <= trained_until:
        st.caption(f"The models were trained on data up to {trained_until:%d %b %Y}, so scores for earlier dates are better than they would be on new data.")


def show_upload_result(pipeline, info: dict) -> None:
    """Summary, scores, download and the usual prediction tables for the latest upload run."""
    path = pipeline.UPLOAD_PREDICTIONS_FILE
    predictions = load_forecasts(str(path), path.stat().st_mtime).sort_values(["forecast_for", "road_id"])
    st.subheader("Latest upload run")
    for box, (label, value) in zip(st.columns(3), [("Rows uploaded", f"{info['rows_uploaded']:,}"), ("Forecasts made", f"{info['rows_predicted']:,}"), ("Roads", info["roads"])]):
        box.metric(label, value)
    models = ", ".join(f"{TARGET_LABELS[target]}: {detail['model']}" for target, detail in info["models"].items())
    st.caption(f"Run {info['run_id']} on {info['created'][:10]}. Forecasts for {info['forecast_from'][:10]} to {info['forecast_to'][:10]}. Models used: {models}. "
               f"Traffic file: {info['files'][pipeline.TRAFFIC_FILE]}. Weather: {info['files'][pipeline.WEATHER_FILE]}. Calendar: {info['files'][pipeline.CALENDAR_FILE]}.")
    upload_scores(predictions, info)
    st.download_button("Download these predictions (CSV)", path.read_bytes(), file_name=path.name, mime="text/csv", key="upload_download")

    prefix = f"upload_{info['run_id']}_"  # a new run starts with fresh pickers, so old choices cannot point at missing dates
    date_box, time_box, road_box = st.columns(3)
    window = pick_window(predictions, date_box, time_box, prefix)
    rows = predictions[predictions["forecast_for"] == window]
    road = pick_road(rows, road_box, prefix)
    st.write(f"forecast_for {window:%A %d %B %Y , %H : %M}")
    corridor_summary(rows)
    segment_table(rows)
    road_detail(road)

    with st.expander("What the cleaning step found and fixed"):
        st.dataframe(pd.DataFrame(list(info["cleaning_summary"].items()), columns=["Check", "Result"]), hide_index=True, width="stretch")
        if info["quarantined_rows"]:
            st.write("Rows set aside (quarantined) instead of being silently dropped:")
            st.dataframe(pd.DataFrame(list(info["quarantined_rows"].items()), columns=["Reason", "Rows"]), hide_index=True)
        else:
            st.caption("No rows were set aside.")


def upload_view() -> None:
    """Upload a new traffic file, run the pipeline on it and look at the predictions."""
    import pipeline  # loaded here because it brings in the model code, which the other tabs do not need

    st.write("Upload a new traffic sensor log. FlowCast checks it, cleans it, builds the features and predicts the next 30 minutes "
             "for every window with the saved models. Nothing is retrained and your original data and predictions are not changed.")
    with st.expander("What the files must look like (templates to download)"):
        st.write("Each template has the exact columns and one example row. Replace the example row with your own rows.")
        notes = {
            pipeline.TRAFFIC_FILE: "Traffic sensor log: date as YYYY-MM-DD, time as HH:MM on the hour or half hour. Needs at least "
                                   f"{pipeline.MIN_DAYS} days of readings for every road, because the features look back one week.",
            pipeline.WEATHER_FILE: "Weather observations: date as DD/MM/YYYY, time on the hour (HH:00). One row per station per hour.",
            pipeline.CALENDAR_FILE: "Calendar events: date as YYYY-MM-DD, flags as 0 or 1. The holiday and event names can stay empty.",
        }
        for name, template in pipeline.TEMPLATE_FILES.items():
            file = pipeline.TEMPLATE_DIRECTORY / template
            note_box, button_box = st.columns([4, 1])
            note_box.write(notes[name])
            if file.exists():
                button_box.download_button("Download template", file.read_bytes(), file_name=file.name, mime="text/csv", key=f"template_{name}")
    traffic = st.file_uploader("Traffic sensor log (CSV)", type="csv", key="upload_traffic")
    with st.expander("Optional: upload weather and calendar files as well"):
        st.caption("If you skip this, the original weather and calendar tables are used. They must cover the dates in your traffic file.")
        weather = st.file_uploader("Weather observations (CSV)", type="csv", key="upload_weather")
        calendar = st.file_uploader("Calendar events (CSV)", type="csv", key="upload_calendar")
    if st.button("Run prediction", type="primary", disabled=traffic is None, key="upload_run"):
        start_upload_run(pipeline, traffic, weather, calendar)

    info = pipeline.load_last_run()
    if info:
        show_upload_result(pipeline, info)
    else:
        st.info("No upload has been run yet.")


def report_view(history: pd.DataFrame) -> None:
    """Build an insights report for a chosen period and offer it as a download."""
    st.write("Pick a period. The report below summarises it and can be downloaded.")
    rows = pick_dates(history, "report_dates")
    if rows.empty:
        st.warning("No data for these dates.")
        return
    start, end = rows["timestamp"].min().date(), rows["timestamp"].max().date()
    forecasts = None
    if PREDICTION_FILE.exists():
        forecasts = load_forecasts(str(PREDICTION_FILE), PREDICTION_FILE.stat().st_mtime)
        days = forecasts["forecast_for"].dt.date
        forecasts = forecasts[(days >= start) & (days <= end)]
    text, roads = insights.build_report(rows, forecasts)
    left, right = st.columns(2)
    left.download_button("Download report (Markdown)", text, file_name=f"flowcast_report_{start}_{end}.md", mime="text/markdown", key="report_md")
    right.download_button("Download road table (CSV)", roads.to_csv().encode("utf-8"), file_name=f"flowcast_roads_{start}_{end}.csv", mime="text/csv", key="report_csv")
    st.divider()
    st.markdown(text)


def main() -> None:
    st.set_page_config(page_title="FlowCast", layout="wide")
    st.title("FlowCast")
    history = load_history(INTERIM_FILE.stat().st_mtime)

    live, trends, heatmap, comparison, performance, importance, forecast, weather, upload, report = st.tabs([
        "Live prediction", "Historical trends", "Congestion heatmap", "Road comparison",
        "Model performance", "Feature importance", "Forecast visualisation",
        "Weather vs traffic", "Upload & predict", "Reports & insights",
    ])
    with live:
        live_view()
    with trends:
        trends_view(history)
    with heatmap:
        heatmap_view(history)
    with comparison:
        comparison_view(history)
    with performance:
        performance_view()
    with importance:
        importance_view()
    with forecast:
        forecast_view()
    with weather:
        weather_view(history)
    with upload:
        upload_view()
    with report:
        report_view(history)




if __name__ == "__main__" :
    main()