"""Analysis helpers shared by the dashboard: weather effects and the exported insights report (M4)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from evaluate import multiclass_metrics, regression_metrics
from ingest import TIMEZONE

RAIN_BANDS = ([-1, 0, 2.5, 7.5, float("inf")], ["No rain", "Light (up to 2.5 mm)", "Moderate (2.5 to 7.5 mm)", "Heavy (over 7.5 mm)"])
VISIBILITY_BANDS = ([-1, 1000, 3000, float("inf")], ["Low (under 1 km)", "Reduced (1 to 3 km)", "Good (over 3 km)"])
CONGESTION_LEVELS = ["Free-flow", "Moderate", "Heavy", "Severe"]
BAD_WEATHER = ["rain", "fog"]


def add_weather_columns(history: pd.DataFrame) -> pd.DataFrame:
    """Add rain and visibility bands, and how far volume and speed are from normal.

    "Normal" is the average for the same road, the same 30 minute slot of the day and the same
    day type (weekday or weekend), so time of day does not hide the weather effect.
    """
    history = history.copy()
    slot = history["timestamp"].dt.hour * 2 + history["timestamp"].dt.minute // 30
    weekend = history["timestamp"].dt.dayofweek >= 5
    for column, name in [("traffic_volume", "volume_vs_typical"), ("avg_speed", "speed_vs_typical")]:
        typical = history.groupby(["road_id", slot, weekend])[column].transform("mean")
        history[name] = history[column] / typical - 1
    history["rain_band"] = pd.cut(history["rainfall"], RAIN_BANDS[0], labels=RAIN_BANDS[1])
    history["visibility_band"] = pd.cut(history["visibility"], VISIBILITY_BANDS[0], labels=VISIBILITY_BANDS[1])
    return history


def weather_effect(history: pd.DataFrame, by: str) -> pd.DataFrame:
    """Volume, speed, incidents and congestion for each value of one weather column."""
    grouped = history.groupby(by, observed=True)
    table = pd.DataFrame({
        "Windows": grouped.size(),
        "Volume vs typical (%)": grouped["volume_vs_typical"].mean() * 100,
        "Speed vs typical (%)": grouped["speed_vs_typical"].mean() * 100,
        "Incidents per 1,000 windows": grouped["accident_count"].mean() * 1000,
        "Heavy or Severe (%)": grouped["congestion_level"].apply(lambda levels: levels.isin(["Heavy", "Severe"]).mean() * 100),
    })
    table.index.name = by
    return table


def forecast_error_by_weather(predictions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """How far the volume forecast is off in each weather condition."""
    weather = history[["road_id", "timestamp", "weather_condition"]].rename(columns={"timestamp": "forecast_for"})
    joined = predictions[["road_id", "forecast_for", "actual_volume", "predicted_volume"]].merge(weather, on=["road_id", "forecast_for"], how="inner")
    joined["percent_error"] = (joined["predicted_volume"] - joined["actual_volume"]).abs() / joined["actual_volume"] * 100
    grouped = joined.groupby("weather_condition")
    table = pd.DataFrame({
        "Windows": grouped.size(),
        "Average actual volume": grouped["actual_volume"].mean(),
        "Average predicted volume": grouped["predicted_volume"].mean(),
        "Forecast error, MAPE (%)": grouped["percent_error"].mean(),
    })
    table.index.name = "weather_condition"
    return table


def markdown_table(table: pd.DataFrame, decimals: int = 1) -> str:
    """A small markdown table without needing any extra library."""
    def cell(value) -> str:
        return f"{value:,.{decimals}f}" if isinstance(value, (float, np.floating)) else str(value)

    header = "| " + " | ".join([table.index.name or ""] + [str(column) for column in table.columns]) + " |"
    divider = "|" + "---|" * (len(table.columns) + 1)
    lines = [header, divider]
    for index, row in table.iterrows():
        lines.append("| " + " | ".join([str(index)] + [cell(value) for value in row]) + " |")
    return "\n".join(lines)


def road_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One line per road for the chosen period."""
    grouped = rows.groupby("road_id")
    table = pd.DataFrame({
        "Road name": grouped["road_name"].first(),
        "Average volume": grouped["traffic_volume"].mean(),
        "Average speed (km/h)": grouped["avg_speed"].mean(),
        "Average travel time (min)": grouped["travel_time"].mean(),
        "Heavy or Severe (%)": grouped["congestion_level"].apply(lambda levels: levels.isin(["Heavy", "Severe"]).mean() * 100),
        "Incidents": grouped["accident_count"].sum(),
        "Travel-time variability (%)": grouped["travel_time"].std() / grouped["travel_time"].mean() * 100,
    })
    table.index.name = "Road"
    return table


def traffic_pattern_lines(rows: pd.DataFrame) -> list[str]:
    """Sentences about the busiest time of day and weekday against weekend."""
    slot = rows["timestamp"].dt.strftime("%H:%M")
    by_slot = rows.groupby(slot)["traffic_volume"].mean()
    weekend = rows["timestamp"].dt.dayofweek >= 5
    lines = [f"- Busiest time of day: {by_slot.idxmax()} (average {by_slot.max():,.0f} vehicles per road per 30 minutes). Quietest: {by_slot.idxmin()} ({by_slot.min():,.0f})."]
    if weekend.any() and (~weekend).any():
        weekday_volume, weekend_volume = rows.loc[~weekend, "traffic_volume"].mean(), rows.loc[weekend, "traffic_volume"].mean()
        lines.append(f"- Weekday average volume is {weekday_volume:,.0f} against {weekend_volume:,.0f} at the weekend ({(weekend_volume / weekday_volume - 1) * 100:+.0f}%).")
    else:
        lines.append("- The period holds only weekdays or only weekend days, so no weekday against weekend comparison is made.")
    return lines


def weather_lines(rows: pd.DataFrame) -> list[str]:
    """Sentences about weather and incidents in the period."""
    bad = rows["weather_condition"].isin(BAD_WEATHER)
    total_incidents = int(rows["accident_count"].sum())
    lines = [f"- Rain or fog was recorded in {bad.mean() * 100:.1f}% of windows."]
    if bad.any():
        lines.append(f"- In those windows volume was {rows.loc[bad, 'volume_vs_typical'].mean() * 100:+.1f}% and speed {rows.loc[bad, 'speed_vs_typical'].mean() * 100:+.1f}% compared with normal for the same road and time of day.")
    lines.append(f"- Incidents reported: {total_incidents}.")
    if bad.any() and total_incidents:
        share = rows.loc[bad, "accident_count"].sum() / total_incidents * 100
        lines.append(f"- {share:.0f}% of the incidents happened in rain or fog, which made up {bad.mean() * 100:.0f}% of windows.")
    return lines


def forecast_lines(predictions: pd.DataFrame) -> list[str]:
    """Sentences about how accurate the forecasts were in the period."""
    volume = regression_metrics(predictions["actual_volume"].to_numpy(dtype=float), predictions["predicted_volume"].to_numpy(dtype=float))
    inside = ((predictions["actual_volume"] >= predictions["predicted_volume_lower"]) & (predictions["actual_volume"] <= predictions["predicted_volume_upper"])).mean()
    congestion = multiclass_metrics(predictions["actual_congestion"].to_numpy(dtype=int), predictions["predicted_congestion"].to_numpy(dtype=int), [0, 1, 2, 3])
    return [
        f"- Forecast windows in this period: {len(predictions):,}.",
        f"- Volume forecast: average error {volume['MAE']:.1f} vehicles (MAPE {volume['MAPE']:.1f}%, RMSE {volume['RMSE']:.1f}).",
        f"- The 90% volume band held the actual value {inside * 100:.0f}% of the time.",
        f"- Congestion level was right {congestion['accuracy'] * 100:.0f}% of the time (macro-F1 {congestion['macro_F1']:.2f}).",
    ]


def build_report(rows: pd.DataFrame, predictions: pd.DataFrame | None) -> tuple[str, pd.DataFrame]:
    """The insights report for the rows given, as markdown text, plus the per-road table."""
    start, end = rows["timestamp"].min(), rows["timestamp"].max()
    days = rows["timestamp"].dt.date.nunique()
    roads = road_table(rows)
    levels = rows["congestion_level"].value_counts(normalize=True).reindex(CONGESTION_LEVELS).fillna(0) * 100

    watch = roads.sort_values("Heavy or Severe (%)", ascending=False).head(5)
    slowest = roads["Average speed (km/h)"].idxmin()
    lines = [
        "# FlowCast insights report",
        "",
        f"**Period:** {start:%d %b %Y} to {end:%d %b %Y} ({days} days)  ",
        f"**Roads:** {rows['road_id'].nunique()} · **Windows:** {len(rows):,} (30 minutes each)  ",
        f"**Created:** {datetime.now(ZoneInfo(TIMEZONE)):%d %b %Y %H:%M}",
        "",
        "## Key numbers",
        "",
        f"- Average volume: {rows['traffic_volume'].mean():,.0f} vehicles per road per window.",
        f"- Average speed: {rows['avg_speed'].mean():.1f} km/h. Average travel time: {rows['travel_time'].mean():.2f} minutes.",
        f"- Slowest road on average: {slowest} at {roads.loc[slowest, 'Average speed (km/h)']:.1f} km/h.",
        "",
        "## Congestion mix",
        "",
        "| Level | Share of windows (%) |",
        "|---|---|",
        *[f"| {level} | {share:.1f} |" for level, share in levels.items()],
        "",
        "## Traffic pattern",
        "",
        *traffic_pattern_lines(rows),
        "",
        "## Roads to watch (highest share of Heavy or Severe windows)",
        "",
        markdown_table(watch[["Road name", "Heavy or Severe (%)", "Average speed (km/h)", "Average travel time (min)", "Incidents"]]),
        "",
        "## Weather and incidents",
        "",
        *weather_lines(rows),
        "",
        "## Forecast accuracy in this period",
        "",
    ]
    if predictions is not None and not predictions.empty:
        lines += forecast_lines(predictions)
    else:
        lines.append("- No forecasts exist for this period. Forecasts cover the validation and test windows only.")
    lines += [
        "",
        "## Where the numbers come from",
        "",
        "- Traffic, speed, congestion, weather and incidents: the cleaned dataset `data/interim/merged.csv`.",
        "- Forecast accuracy: `data/processed/predictions_v1.csv` (winning classical models).",
        "- Volume and speed changes are compared with the average for the same road, time of day and day type.",
    ]
    return "\n".join(lines) + "\n", roads
