import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
QUARANTINE_DIRECTORY = PROJECT_ROOT / "data" / "quarantine"

TRAFFIC_FILE = "traffic_sensor_log.csv"
WEATHER_FILE = "weather_observations.csv"
CALENDAR_FILE = "calendar_events.csv"

REQUIRED_COLUMNS = {
    TRAFFIC_FILE: [
        "road_id", "road_name", "latitude", "longitude", "weather_station_id",
        "date", "time", "traffic_volume", "vehicle_count", "vehicle_type_dist",
        "avg_speed", "occupancy", "congestion_level", "travel_time",
        "accident_count", "signal_timing", "road_capacity",
    ],
    WEATHER_FILE: [
        "station_id", "date", "time", "weather_condition",
        "temperature", "rainfall", "visibility",
    ],
    CALENDAR_FILE: [
        "date", "public_holiday", "holiday_name", "event_flag",
        "event_name", "roadwork_flag",
    ],
}

NUMERIC_COLUMNS = {
    TRAFFIC_FILE: [
        "latitude", "longitude", "traffic_volume", "vehicle_count", "avg_speed",
        "occupancy", "travel_time", "accident_count", "signal_timing",
        "road_capacity",
    ],
    WEATHER_FILE: ["temperature", "rainfall", "visibility"],
    CALENDAR_FILE: ["public_holiday", "event_flag", "roadwork_flag"],
}

KEY_COLUMNS = {
    TRAFFIC_FILE: ["road_id", "timestamp"],
    WEATHER_FILE: ["station_id", "timestamp"],
}

TIMESTAMP_FORMATS = {
    TRAFFIC_FILE: "%Y-%m-%d %H:%M",
    WEATHER_FILE: "%d/%m/%Y %H:%M",
}
ALLOWED_MINUTES = {
    TRAFFIC_FILE: [0, 30],
    WEATHER_FILE: [0],
}
# timezone is assumed from the coordinates, the PRD does not give one
TIMEZONE = "Asia/Kolkata"

VALUE_LIMITS = {
    "latitude": (-90, 90),
    "longitude": (-180, 180),
    "traffic_volume": (0, None),
    "vehicle_count": (0, None),
    "avg_speed": (0, 200),
    "occupancy": (0, 100),
    "travel_time": (0, None),
    "accident_count": (0, None),
    "signal_timing": (0, None),
    "road_capacity": (1, None),
    "temperature": (-60, 60),
    "rainfall": (0, None),
    "visibility": (0, None),
}

MISSING_MARKERS = ["", "na", "n/a", "nan", "null", "none", "-", "?"]


def quarantine_rows(rows: pd.DataFrame, dataset_name: str, reason: str) -> None:
    """Save rejected rows to data/quarantine."""
    path = QUARANTINE_DIRECTORY / f"{Path(dataset_name).stem}__{reason}.csv"
    if rows.empty:
        path.unlink(missing_ok=True)
        return
    QUARANTINE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)
    logger.warning("%s: %d rows quarantined (%s)", dataset_name, len(rows), reason)


def load_csv(name: str) -> pd.DataFrame:
    """Load one raw csv file."""
    path = RAW_DATA_DIRECTORY / name
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found : {name}")
    string_columns = ["road_id", "weather_station_id", "station_id", "date", "time"]
    data = pd.read_csv(path, dtype={column: "string" for column in string_columns})
    if data.empty:
        raise ValueError(f"{name} has no rows")
    logger.info("Loaded %s: %d rows, %d columns", name, *data.shape)
    return data


def inspect_dataframe(name: str, data: pd.DataFrame) -> None:
    """Print shape, columns, types, missing values and first rows."""
    print(f"\n{'=' * 50}")
    print(f"Dataset : {name}")
    print(f"{'=' * 50}")
    print(f"Rows : {data.shape[0]}")
    print(f"Columns : {data.shape[1]}")
    print("\nColumn names :")
    print(data.columns.to_list())
    print("\nData types :")
    print(data.dtypes)
    print("\nMissing values per column :")
    print(data.isnull().sum())
    print("\nFirst 5 rows :")
    print(data.head())


def validate_required_columns(data: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """Raise an error if a required column is missing."""
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {dataset_name} : {missing_columns}")
    print(f"{dataset_name} : required columns are validated")


def validate_dataset_schema(data: pd.DataFrame, dataset_name: str) -> None:
    """Check the dataset has all the columns its schema needs."""
    if dataset_name not in REQUIRED_COLUMNS:
        raise ValueError(f"No schema definition found for {dataset_name}")
    validate_required_columns(data, REQUIRED_COLUMNS[dataset_name], dataset_name)


def validate_numeric_types(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Convert numeric columns and quarantine rows that hold text."""
    data = data.copy()
    invalid_rows = pd.Series(False, index=data.index)
    for column in NUMERIC_COLUMNS[dataset_name]:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_rows |= data[column].notna() & converted.isna()
        data[column] = converted
    quarantine_rows(data[invalid_rows], dataset_name, "non_numeric_value")
    return data.loc[~invalid_rows].reset_index(drop=True)


def create_time_stamp(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Combine date and time into one tz-aware timestamp column."""
    data = data.copy()
    date_time_values = data["date"].str.strip() + " " + data["time"].str.strip()
    timestamps = pd.to_datetime(
        date_time_values, format=TIMESTAMP_FORMATS[dataset_name], errors="coerce"
    )

    invalid = timestamps.isna()
    misaligned = ~invalid & ~timestamps.dt.minute.isin(ALLOWED_MINUTES[dataset_name])
    quarantine_rows(data[invalid], dataset_name, "invalid_timestamp")
    quarantine_rows(data[misaligned], dataset_name, "misaligned_timestamp")

    keep = ~(invalid | misaligned)
    data = data.loc[keep].copy()
    data["timestamp"] = timestamps[keep].dt.tz_localize(TIMEZONE)
    print(f"Timestamp creation : {dataset_name}")
    print(f"Invalid time stamps : {invalid.sum()}")
    return data.reset_index(drop=True)


def validate_station_ids(traffic: pd.DataFrame, weather: pd.DataFrame) -> None:
    """Make sure every station used in traffic exists in weather."""
    unknown = set(traffic["weather_station_id"].dropna()) - set(weather["station_id"])
    if unknown:
        raise ValueError(f"Stations missing from weather data : {sorted(unknown)}")


def inspect_duplicate_keys(data: pd.DataFrame, dataset_name: str) -> int:
    """Print how many rows share the same key."""
    keys = KEY_COLUMNS.get(dataset_name)
    if keys is None:
        return 0
    duplicate_rows = data.loc[data.duplicated(subset=keys, keep=False)]
    groups = len(duplicate_rows.drop_duplicates(subset=keys))
    print(f"\nDuplicate-key analysis: {dataset_name}")
    print("Rows involved in duplicate keys:", len(duplicate_rows))
    print("Number of duplicate key groups:", groups)
    if not duplicate_rows.empty:
        print(duplicate_rows.sort_values(keys).head(10))
    return groups


def inspect_numeric_ranges(data: pd.DataFrame, dataset_name: str) -> dict[str, int]:
    """Count values outside the valid range of each numeric column."""
    report = {}
    print(f"\nNumeric range analysis: {dataset_name}")
    for column, (low, high) in VALUE_LIMITS.items():
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        below = int((values < low).sum()) if low is not None else 0
        above = int((values > high).sum()) if high is not None else 0
        report[column] = below + above
        print(f"{column}: below {low} = {below}, above {high} = {above}")
    return report


def inspect_missing_values(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Count missing values, including text like N/A or -."""
    missing_count = data.isnull().sum()
    for column in data.columns:
        if pd.api.types.is_string_dtype(data[column]):
            text = data[column].astype("string").str.strip().str.lower()
            missing_count[column] += int(text.isin(MISSING_MARKERS).sum())

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": (missing_count / len(data) * 100).round(2),
    })
    missing_report = missing_report[missing_report["missing_count"] > 0]

    print(f"\nMissing-value analysis: {dataset_name}")
    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report.sort_values(by="missing_count", ascending=False))
    return missing_report


def inspect_missing_windows(traffic: pd.DataFrame) -> int:
    """Count road and 30 minute window combinations that have no row."""
    windows = pd.date_range(traffic["timestamp"].min(), traffic["timestamp"].max(), freq="30min")
    roads = traffic["road_id"].nunique()
    present = len(traffic.drop_duplicates(subset=["road_id", "timestamp"]))
    missing = roads * len(windows) - present
    print(f"\nMissing windows : {missing} of {roads * len(windows)} expected rows")
    return missing


def load_and_validate_tables() -> dict[str, pd.DataFrame]:
    """Load the three raw tables, validate them and add timestamps."""
    tables = {}
    for name in REQUIRED_COLUMNS:
        data = load_csv(name)
        validate_dataset_schema(data, name)
        data = validate_numeric_types(data, name)
        if name in TIMESTAMP_FORMATS:
            data = create_time_stamp(data, name)
        tables[name] = data
    validate_station_ids(tables[TRAFFIC_FILE], tables[WEATHER_FILE])
    return tables


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tables = load_and_validate_tables()
    for name, data in tables.items():
        inspect_dataframe(name, data)
        inspect_missing_values(data, name)
        inspect_duplicate_keys(data, name)
        inspect_numeric_ranges(data, name)
    inspect_missing_windows(tables[TRAFFIC_FILE])


if __name__ == "__main__":
    main()
