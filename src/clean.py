import logging

import pandas as pd

from ingest import (
    CALENDAR_FILE,
    PROJECT_ROOT,
    TRAFFIC_FILE,
    WEATHER_FILE,
    inspect_dataframe,
    inspect_duplicate_keys,
    inspect_missing_values,
    inspect_missing_windows,
    inspect_numeric_ranges,
    load_and_validate_tables,
    quarantine_rows,
)

INTERIM_DIRECTORY = PROJECT_ROOT / "data" / "interim"


def remove_exact_duplicates(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Drop rows that are exactly the same."""
    original_count = len(df)
    df = df.drop_duplicates().copy()
    removed_count = original_count - len(df)

    print(f"\nExact duplicate removal: {dataset_name}")
    print(f"Rows before removal: {original_count}")
    print(f"Exact duplicate rows removed: {removed_count}")
    print(f"Rows after removal: {len(df)}")
    return df


def standardize_weather_conditions(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the weather condition labels."""
    data = data.copy()
    print("Weather condition before standardization :")
    print(data["weather_condition"].value_counts(dropna=False))
    data["weather_condition"] = data["weather_condition"].str.strip().str.lower()
    data["weather_condition"] = data["weather_condition"].replace({"rainy": "rain", "foggy": "fog"})
    print("Weather condition after standardization :")
    print(data["weather_condition"].value_counts(dropna=False))
    return data


def remove_duplicate_traffic_records(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most complete record for each road and timestamp."""
    initial_rows = len(df)
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)
    df = (
        df.sort_values("_completeness", ascending=False, kind="stable")
        .drop_duplicates(subset=["road_id", "timestamp"], keep="first")
        .drop(columns=["_completeness"])
        .reset_index(drop=True)
    )

    print("\n[Traffic] Duplicate records removal")
    print(f"Rows before: {initial_rows}")
    print(f"Duplicate records removed: {initial_rows - len(df)}")
    print(f"Rows after: {len(df)}")
    return df


def remove_duplicate_keys(df: pd.DataFrame, key_columns: list[str], dataset_name: str) -> pd.DataFrame:
    """Keep the most complete row for each key."""
    initial_rows = len(df)
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)
    df = (
        df.sort_values("_completeness", ascending=False, kind="stable")
        .drop_duplicates(subset=key_columns, keep="first")
        .drop(columns=["_completeness"])
        .sort_index()
        .reset_index(drop=True)
    )
    print(f"\n[{dataset_name}] Duplicate keys {key_columns} removed: {initial_rows - len(df)}")
    return df


def handle_invalid_numeric_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Set impossible numeric values to NaN."""
    df = df.copy()
    invalid_counts = {}

    non_negative_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "road_capacity",
        "rainfall",
        "visibility",
    ]
    for column in non_negative_columns:
        if column in df.columns:
            invalid_mask = df[column] < 0
            invalid_counts[column] = int(invalid_mask.sum())
            df.loc[invalid_mask, column] = float("nan")

    if "avg_speed" in df.columns:
        invalid_mask = df["avg_speed"] > 200
        invalid_counts["avg_speed_above_200"] = int(invalid_mask.sum())
        df.loc[invalid_mask, "avg_speed"] = float("nan")

    if "occupancy" in df.columns:
        invalid_mask = df["occupancy"] > 100
        invalid_counts["occupancy_above_100"] = int(invalid_mask.sum())
        df.loc[invalid_mask, "occupancy"] = float("nan")

    print(f"\n[{dataset_name}] Invalid numeric values set to NaN:")
    print(invalid_counts)
    return df


def fill_gaps(df: pd.DataFrame, group_column: str, columns: list[str], max_gap: int, window_key: pd.Series) -> pd.DataFrame:
    """Interpolate short gaps, then use the group median for the rest."""
    indexed = df.set_index("timestamp")
    interpolated = indexed.groupby(group_column)[columns].transform(
        lambda s: s.interpolate(method="time", limit=max_gap, limit_area="inside")
    )
    df[columns] = interpolated.to_numpy()

    for column in columns:
        window_median = df.groupby([group_column, window_key])[column].transform("median")
        df[column] = df[column].fillna(window_median)
    return df


def handle_missing_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Fill missing traffic readings and quarantine rows that cannot be recovered."""
    sensor_columns = ["traffic_volume", "vehicle_count", "avg_speed", "occupancy", "travel_time"]

    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    before = df[sensor_columns].isna().sum()

    time_of_day = df["timestamp"].dt.strftime("%H:%M")
    df = fill_gaps(df, "road_id", sensor_columns, 4, time_of_day)

    unrecoverable = df[sensor_columns].isna().any(axis=1)
    quarantine_rows(df[unrecoverable], dataset_name, "unrecoverable_missing")
    df = df.loc[~unrecoverable].reset_index(drop=True)

    print(f"\n[{dataset_name}] Missing values handled")
    print("Missing before :")
    print(before)
    print("Missing after :")
    print(df[sensor_columns].isna().sum())
    print(f"Rows quarantined : {int(unrecoverable.sum())}")
    return df


def handle_missing_weather(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Fill missing weather values for each station."""
    weather_columns = ["temperature", "rainfall", "visibility"]

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    before = df[weather_columns].isna().sum()

    df = fill_gaps(df, "station_id", weather_columns, 3, df["timestamp"].dt.hour)

    print(f"\n[{dataset_name}] Missing weather values handled")
    print("Missing before :")
    print(before)
    print("Missing after :")
    print(df[weather_columns].isna().sum())
    return df


def derive_congestion_level(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Fill missing congestion levels from the volume to capacity ratio."""
    df = df.copy()
    ratio = df["traffic_volume"] / df["road_capacity"]
    derived = pd.cut(
        ratio,
        bins=[-float("inf"), 0.25, 0.40, 0.50, float("inf")],
        labels=["Free-flow", "Moderate", "Heavy", "Severe"],
        right=False,
    ).astype("string")

    has_level = df["congestion_level"].notna()
    mismatch = int((derived[has_level] != df.loc[has_level, "congestion_level"]).sum())
    missing_before = int((~has_level).sum())
    df["congestion_level"] = df["congestion_level"].fillna(derived)

    print(f"\n[{dataset_name}] Congestion level derived")
    print(f"Missing before: {missing_before}")
    print(f"Missing after: {int(df['congestion_level'].isna().sum())}")
    print(f"Existing labels that disagree with the rule: {mismatch} of {int(has_level.sum())}")
    print(df["congestion_level"].value_counts())
    return df


def convert_integer_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Convert the count columns back to whole numbers."""
    df = df.copy()
    integer_columns = ["traffic_volume", "vehicle_count", "accident_count", "signal_timing", "road_capacity"]
    for column in integer_columns:
        df[column] = df[column].round().astype(int)

    print(f"\n[{dataset_name}] Converted to integer : {integer_columns}")
    return df


def merge_datasets(traffic_df: pd.DataFrame, weather_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Join weather and calendar onto the traffic records."""
    traffic_df = traffic_df.copy()
    weather_df = weather_df.drop(columns=["date", "time"]).rename(columns={"timestamp": "weather_hour"})
    calendar_df = calendar_df.copy()

    traffic_df["weather_hour"] = traffic_df["timestamp"].dt.floor("h")
    traffic_df["calendar_date"] = traffic_df["timestamp"].dt.date
    calendar_df["calendar_date"] = pd.to_datetime(calendar_df["date"]).dt.date
    calendar_df = calendar_df.drop(columns=["date"])

    merged_df = traffic_df.merge(
        weather_df,
        left_on=["weather_station_id", "weather_hour"],
        right_on=["station_id", "weather_hour"],
        how="left",
        validate="many_to_one",
        indicator="_weather_match",
    )
    merged_df = merged_df.merge(
        calendar_df,
        on="calendar_date",
        how="left",
        validate="many_to_one",
        indicator="_calendar_match",
    )

    if len(merged_df) != len(traffic_df):
        raise ValueError(f"Merge changed the row count: {len(traffic_df)} -> {len(merged_df)}")

    no_weather = int((merged_df["_weather_match"] == "left_only").sum())
    no_calendar = int((merged_df["_calendar_match"] == "left_only").sum())
    merged_df = merged_df.drop(columns=[
        "_weather_match", "_calendar_match", "station_id",
        "weather_station_id", "weather_hour", "calendar_date",
    ])

    print("\nDataset merge completed.")
    print(f"Traffic rows: {len(traffic_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged columns: {len(merged_df.columns)}")
    print(f"Rows with no weather: {no_weather}")
    print(f"Rows with no calendar date: {no_calendar}")
    return merged_df


def inspect_merged_dataset(merged_data: pd.DataFrame) -> None:
    """Print the checks for the merged dataset."""
    print("===============================inspect merged dataset ==================================================================")
    inspect_dataframe("merged dataset", merged_data)
    inspect_numeric_ranges(merged_data, "merged dataset")

    duplicate_count = int(merged_data.duplicated(subset=["road_id", "timestamp"]).sum())
    print("\nMerged dataset checks")
    print(f"Roads: {merged_data['road_id'].nunique()}")
    print(f"Time range: {merged_data['timestamp'].min()} to {merged_data['timestamp'].max()}")
    print(f"Duplicate road/timestamp rows: {duplicate_count}")
    print("\nCongestion levels:")
    print(merged_data["congestion_level"].value_counts(dropna=False))
    print("\nWeather conditions:")
    print(merged_data["weather_condition"].value_counts(dropna=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tables = load_and_validate_tables()

    for name in tables:
        tables[name] = remove_exact_duplicates(tables[name], name)

    for name, data in tables.items():
        inspect_missing_values(data, name)
        inspect_duplicate_keys(data, name)
        inspect_numeric_ranges(data, name)
    inspect_missing_windows(tables[TRAFFIC_FILE])

    traffic = remove_duplicate_traffic_records(tables[TRAFFIC_FILE])
    weather = remove_duplicate_keys(tables[WEATHER_FILE], ["station_id", "timestamp"], WEATHER_FILE)
    calendar = remove_duplicate_keys(tables[CALENDAR_FILE], ["date"], CALENDAR_FILE)

    weather = standardize_weather_conditions(weather)
    traffic = handle_invalid_numeric_values(traffic, TRAFFIC_FILE)
    weather = handle_invalid_numeric_values(weather, WEATHER_FILE)

    traffic = handle_missing_values(traffic, TRAFFIC_FILE)
    weather = handle_missing_weather(weather, WEATHER_FILE)
    traffic = derive_congestion_level(traffic, TRAFFIC_FILE)
    traffic = convert_integer_columns(traffic, TRAFFIC_FILE)

    merged_data = merge_datasets(traffic, weather, calendar)
    inspect_merged_dataset(merged_data)

    INTERIM_DIRECTORY.mkdir(parents=True, exist_ok=True)
    merged_data.to_csv(INTERIM_DIRECTORY / "merged.csv", index=False)


if __name__ == "__main__":
    main()
