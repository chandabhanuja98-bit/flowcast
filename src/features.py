import pandas as pd
import numpy as np
from ingest import PROJECT_ROOT,TIMEZONE
import json 
from datetime import datetime, timezone

REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality.md"
DATASET_VERSION = "v1"
OPTIONAL_COLUMNS = ["vehicle_type_dist", "holiday_name", "event_name"]
PEAK_HOURS = [(7,10),(16,20)]
LAG_WINDOW = [1,2,48,336]
ROLLING_WINDOWS = [4,8]
LOW_VISIBILITY = 1000
CONGESTION_ORDER = {"Free-flow": 0, "Moderate": 1, "Heavy": 2, "Severe": 3}
TEMPERATURE_BANDS = [-float("inf"), 15, 22, float("inf")]
INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "merged.csv"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMNS = ["target_volume", "target_travel_time", "target_congestion", "target_accident"]


def create_temporial_features (df : pd.DataFrame) -> pd.DataFrame :

    df = df.copy()
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60
    day = df["timestamp"].dt.day_of_week 
    df["hour_sin"] = np.sin(2*np.pi*hour/24)
    df["hour_cos"] = np.cos(2*np.pi*hour/24)

    df["day_sin"] = np.sin(2*np.pi*day/7)
    df["day_cos"] = np.cos(2*np.pi*day/7)

    df["is_weekend"] = (day >= 5).astype(int)
    is_peak = pd.Series(False,index=df.index)

    for start,end in PEAK_HOURS :
        is_peak |= (hour>=start) & (hour<end)
    df["is_peak"] = is_peak.astype(int)

    return df

def create_lag_features (df : pd.DataFrame) -> pd.DataFrame :
    df = df.sort_values(["road_id","timestamp"]).copy()
    for lag in LAG_WINDOW :
        df[f"volume_lag_{lag}"] = df.groupby("road_id")["traffic_volume"].shift(lag)  
        df[f"speed_lag_{lag}"] = df.groupby("road_id")["avg_speed"].shift(lag) 

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.sort_values(["road_id", "timestamp"]).copy()
    for window in ROLLING_WINDOWS:
        for column, name in [("traffic_volume", "volume"), ("avg_speed", "speed")]:
            grouped = df.groupby("road_id")[column] 
            df[f"{name}_roll_mean_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window).mean())
            df[f"{name}_roll_std_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window).std())
    return df

def create_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume_capacity_ratio"] = df["traffic_volume"] / df["road_capacity"]
    df["headroom"] = df["road_capacity"] - df["traffic_volume"]
    return df


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume_change_1"] = df["traffic_volume"] - df["volume_lag_1"]
    df["volume_change_2"] = df["volume_lag_1"] - df["volume_lag_2"]
    df["speed_change_1"] = df["avg_speed"] - df["speed_lag_1"]
    df["ratio_change_1"] = df["volume_change_1"] / df["road_capacity"]
    df["volume_vs_mean_4"] = df["traffic_volume"] - df["volume_roll_mean_4"]
    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["road_id", "timestamp"]).copy()
    slot = df["timestamp"].dt.hour * 2 + df["timestamp"].dt.minute // 30
    df["slot_mean_volume"] = df.groupby(["road_id", slot])["traffic_volume"].transform(lambda s: s.shift(1).expanding().mean())

    by_road = df.groupby("road_id")
    df["next_slot_mean_volume"] = by_road["slot_mean_volume"].shift(-1)
    df["next_slot_mean_ratio"] = df["next_slot_mean_volume"] / df["road_capacity"]
    df["volume_vs_next_slot_mean"] = df["traffic_volume"] - df["next_slot_mean_volume"]
    for column in ["hour_sin", "hour_cos", "is_peak", "is_weekend"]:
        df[f"next_{column}"] = by_road[column].shift(-1)
    return df


def create_network_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_time = df.groupby("timestamp")
    df["network_mean_ratio"] = by_time["volume_capacity_ratio"].transform("mean")
    df["network_mean_speed"] = by_time["avg_speed"].transform("mean")
    df["ratio_vs_network"] = df["volume_capacity_ratio"] - df["network_mean_ratio"]
    return df

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_rain"] = (df["rainfall"] > 0).astype(int)
    df["is_low_visibility"] = (df["visibility"] < LOW_VISIBILITY).astype(int)
    df["temperature_band"] = pd.cut(
        df["temperature"], bins=TEMPERATURE_BANDS, labels=[0, 1, 2], right=False
    ).astype(int)
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["congestion_code"] = df["congestion_level"].map(CONGESTION_ORDER)
    weather = pd.get_dummies(df["weather_condition"], prefix="weather", dtype=int)
    return pd.concat([df, weather], axis=1)

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["holiday_peak"] = df["public_holiday"] * df["is_peak"]

    day = df["timestamp"].dt.date
    daily_event = df.groupby(day)["event_flag"].max()
    df["event_yesterday"] = day.map(daily_event.shift(1)).fillna(0).astype(int)
    df["event_tomorrow"] = day.map(daily_event.shift(-1)).fillna(0).astype(int)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = create_temporial_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_capacity_features(df)
    df = create_trend_features(df)
    df = create_seasonal_features(df)
    df = create_network_features(df)
    df = create_weather_features(df)
    df = encode_categorical_features(df)
    df = create_calendar_features(df)
    df = create_targets(df)
    return df

def validate_features(df: pd.DataFrame) -> None:
    modelling_columns = [column for column in df.columns if column not in OPTIONAL_COLUMNS]
    null_counts = df[modelling_columns].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Nulls found in modelling columns: {null_counts.to_dict()}")

    if df.duplicated(subset=["road_id", "timestamp"]).any():
        raise ValueError("Duplicate road/timestamp rows found")

    if df.groupby("road_id").size().nunique() != 1:
        raise ValueError("Roads have a different number of rows")
    print("Feature validation passed")

def save_processed_dataset(df: pd.DataFrame, original_columns: list[str], rows_dropped: int) -> dict:
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data_file = PROCESSED_DIRECTORY / f"features_{DATASET_VERSION}.csv"
    df.to_csv(data_file, index=False)

    info = {
        "version": DATASET_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(INTERIM_FILE.relative_to(PROJECT_ROOT)),
        "rows": len(df),
        "columns": len(df.columns),
        "roads": int(df["road_id"].nunique()),
        "time_from": str(df["timestamp"].min()),
        "time_to": str(df["timestamp"].max()),
        "rows_dropped_no_history": rows_dropped,
        "feature_columns": [column for column in df.columns if column not in original_columns and column not in TARGET_COLUMNS], 
        "target_columns" : TARGET_COLUMNS
    }
    info_file = PROCESSED_DIRECTORY / f"features_{DATASET_VERSION}.json"
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved {data_file.name} and {info_file.name}")
    return info

def update_data_quality_report(info: dict) -> None:
    report = REPORT_FILE.read_text(encoding="utf-8").split("## Processed dataset")[0].rstrip()
    lines = [
        report,
        "",
        "## Processed dataset",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Version | {info['version']} |",
        f"| Source | {info['source']} |",
        f"| Rows | {info['rows']} |",
        f"| Columns | {info['columns']} ({len(info['feature_columns'])} engineered), {len(info['target_columns'])} targets | ",
        f"| Targets | {','.join(info['target_columns'])} | ",
        f"| Roads | {info['roads']} |",
        f"| Time range | {info['time_from']} to {info['time_to']} |",
        f"| Rows dropped (no lag or rolling history yet) | {info['rows_dropped_no_history']} |",
        "| Nulls in modelling columns | 0 |",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["road_id", "timestamp"]).copy()
    grouped = df.groupby("road_id")
    df["target_volume"] = grouped["traffic_volume"].shift(-1)
    df["target_travel_time"] = grouped["travel_time"].shift(-1)
    df["target_congestion"] = grouped["congestion_code"].shift(-1)
    next_accidents = grouped["accident_count"].shift(-1)
    df["target_accident"] = (next_accidents > 0).astype(float).where(next_accidents.notna())
    return df

def main() -> None:
    df = pd.read_csv(INTERIM_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    original_columns = df.columns.to_list()
    df = build_features(df)
    
    history_columns = [column for column in df.columns if "_lag_" in column or "_roll_" in column]
    rows_before = len(df)
    df = df.dropna(subset=history_columns+TARGET_COLUMNS).reset_index(drop=True)
    for column in ["target_volume","target_congestion","target_accident"] :
        df[column] = df[column].astype(int)
    print(f"Rows before: {rows_before}")
    print(f"Rows dropped (no history yet): {rows_before - len(df)}")
    print(f"Rows after: {len(df)}")
    print(f"Columns: {len(df.columns)}")


    validate_features(df)
    info = save_processed_dataset(df,original_columns,rows_before-len(df))
    update_data_quality_report(info)


if __name__ == "__main__":
    main()





