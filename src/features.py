import pandas as pd
import numpy as np
from ingest import PROJECT_ROOT,TIMEZONE

PEAK_HOURS = [(7,10),(14,19)]
LAG_WINDOW = [1,2,48]
ROLLING_WINDOWS = [4,8]
LOW_VISIBILITY = 1000
CONGESTION_ORDER = {"Free-flow": 0, "Moderate": 1, "Heavy": 2, "Severe": 3}
TEMPERATURE_BANDS = [-float("inf"), 15, 22, float("inf")]
INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "merged.csv"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

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
        is_peak = (hour>=start) & (hour<=end)
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
    df = create_weather_features(df)
    df = encode_categorical_features(df)
    df = create_calendar_features(df)
    return df

def main() -> None:
    df = pd.read_csv(INTERIM_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    df = build_features(df)

    history_columns = [column for column in df.columns if "_lag_" in column or "_roll_" in column]
    rows_before = len(df)
    df = df.dropna(subset=history_columns).reset_index(drop=True)
    print(f"Rows before: {rows_before}")
    print(f"Rows dropped (no history yet): {rows_before - len(df)}")
    print(f"Rows after: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIRECTORY / "features.csv", index=False)


if __name__ == "__main__":
    main()





