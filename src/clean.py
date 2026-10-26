import pandas as pd



def remove_exact_duplicates(df: pd.DataFrame, dataset_name: str,) -> pd.DataFrame:
   
    original_count = len(df)

    df = df.drop_duplicates().copy()

    removed_count = original_count - len(df)

    print(f"\nExact duplicate removal: {dataset_name}")
    print(f"Rows before removal: {original_count}")
    print(f"Exact duplicate rows removed: {removed_count}")
    print(f"Rows after removal: {len(df)}")

    return df

def standardize_weather_conditions(data : pd.DataFrame) -> pd.DataFrame :
    data = data.copy()
    print("-------------------------------------Weather condition before standardization --------------------------------")
    print(data["weather_condition"].value_counts(dropna=False))
    data["weather_condition"] = (data["weather_condition"].str.strip().str.lower())
    print("-------------------------------------Weather condition After standardization --------------------------------")
    print(data["weather_condition"].value_counts(dropna=False))
    return data

def remove_duplicate_traffic_records(df: pd.DataFrame) -> pd.DataFrame:

    required_columns = ["road_id", "timestamp"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' is missing from traffic data.")
    initial_rows = len(df)
    # Count non-missing values in each row
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)

    # Keep the most complete record for each road and timestamp
    df = (
        df.sort_values("_completeness", ascending=False)
        .drop_duplicates(subset=["road_id", "timestamp"], keep="first")
        .drop(columns=["_completeness"])
        .reset_index(drop=True)
    )

    removed_rows = initial_rows - len(df)

    print("\n[Traffic] Duplicate records removal")
    print(f"Rows before: {initial_rows}")
    print(f"Duplicate records removed: {removed_rows}")
    print(f"Rows after: {len(df)}")

    return df

def handle_invalid_numeric_values(df: pd.DataFrame,dataset_name: str) -> pd.DataFrame:
    
    df = df.copy()
    invalid_counts = {}

    # Columns that should not contain negative values
    non_negative_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "road_capacity",
        "temperature",
        "rainfall",
        "visibility",
    ]

    for column in non_negative_columns:
        if column in df.columns:
            invalid_mask = df[column] < 0
            invalid_counts[column] = int(invalid_mask.sum())
            df.loc[invalid_mask, column] = float("nan")

    # Average speed above 200 is considered invalid
    if "avg_speed" in df.columns:
        invalid_mask = df["avg_speed"] > 200
        invalid_counts["avg_speed_above_200"] = int(invalid_mask.sum())
        df.loc[invalid_mask, "avg_speed"] = float("nan")

    # Occupancy must not exceed 100%
    if "occupancy" in df.columns:
        invalid_mask = df["occupancy"] > 100
        invalid_counts["occupancy_above_100"] = int(invalid_mask.sum())
        df.loc[invalid_mask, "occupancy"] = float("nan")

    print(f"\n[{dataset_name}] Invalid numeric values handled successfully:")
    print("Average speed after handle : ")
    if "avg_speed" in df.columns:
            invalid_mask = df["avg_speed"] > 200
            invalid_counts["avg_speed_above_200"] = int(invalid_mask.sum())
            print(invalid_counts)
    
    return df

def merge_datasets(
    traffic_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:

    traffic_df = traffic_df.copy()
    weather_df = weather_df.copy()
    calendar_df = calendar_df.copy()

    # Validate required columns
    required_columns = {
        "traffic": ["road_id", "timestamp", "weather_station_id"],
        "weather": ["station_id", "timestamp"],
        "calendar": ["date"],
    }

    for column in required_columns["traffic"]:
        if column not in traffic_df.columns:
            raise ValueError(f"Missing traffic column: {column}")

    for column in required_columns["weather"]:
        if column not in weather_df.columns:
            raise ValueError(f"Missing weather column: {column}")

    for column in required_columns["calendar"]:
        if column not in calendar_df.columns:
            raise ValueError(f"Missing calendar column: {column}")

    # Merge weather observations with traffic records
    merged_df = traffic_df.merge(
        weather_df,
        left_on=["weather_station_id", "timestamp"],
        right_on=["station_id", "timestamp"],
        how="left",
        suffixes=("", "_weather"),
        validate="many_to_one",
    )

    # Prepare calendar date for merging
    merged_df["date"] = pd.to_datetime(
        merged_df["timestamp"],
        errors="coerce",
    ).dt.date

    calendar_df["date"] = pd.to_datetime(
        calendar_df["date"],
        errors="coerce",
    ).dt.date

    # Merge calendar information
    merged_df = merged_df.merge(
        calendar_df,
        on="date",
        how="left",
        suffixes=("", "_calendar"),
        validate="many_to_one",
    )

    print("\nDataset merge completed.")
    print(f"Traffic rows: {len(traffic_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged columns: {len(merged_df.columns)}")

    return merged_df




        


