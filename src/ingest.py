import pandas as pd
from pathlib import Path
from clean import remove_exact_duplicates,standardize_weather_conditions, remove_duplicate_traffic_records, handle_invalid_numeric_values, merge_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent  
RAW_DATA_DIRECTORY = PROJECT_ROOT/"data"/"raw"



def load_csv(name : str) -> pd.DataFrame :
    path = RAW_DATA_DIRECTORY/name
    if not path.exists() :
        raise FileNotFoundError(f"CSV file not found : {name}")
    data = pd.read_csv(path) 
    return data

def inspect_dataframe(name : str, data : pd.DataFrame) -> None :
    print(f"\n {'='*50}") 
    print(f"Dataset : {name}")
    print(f"\n {'='*50}") 
    
    print(f"Rows : {data.shape[0]}")
    print(f"Columns : {data.shape[1]}")
    
    print("\n Coulmns names : ")
    print(data.columns.to_list())

    print("\n Data Types : ")
    print(data.dtypes)

    print("\n Missing values per coulumns : ")
    print(data.isnull().sum())

    print("\n Print first 5 rows : ")
    print(data.head())


def validate_required_columns(data: pd.DataFrame,required_columns : list[str],dataset_name : str) -> None:
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}"
            f"{missing_columns}"
        )
    print(f"{dataset_name} : required columns are validated")

def validate_dataset_schema(data : pd.DataFrame, dataset_name : str) -> None :
    required_columns_by_file = {
        "traffic_sensor_log.csv": [
            "road_id",
            "road_name",
            "latitude",
            "longitude",
            "weather_station_id",
            "date",
            "time",
            "traffic_volume",
            "vehicle_count",
            "vehicle_type_dist",
            "avg_speed",
            "occupancy",
            "congestion_level",
            "travel_time",
            "accident_count",
            "signal_timing",
            "road_capacity",
        ],

        "weather_observations.csv": [
            "station_id",
            "date",
            "time",
            "weather_condition",
            "temperature",
            "rainfall",
            "visibility",
        ],

        "calendar_events.csv": [
            "date",
            "public_holiday",
            "holiday_name",
            "event_flag",
            "event_name",
            "roadwork_flag",
        ],
    }
    if dataset_name not in required_columns_by_file:
        raise ValueError(
            f"No schema definition found for {dataset_name}"
        )
    required_columns = required_columns_by_file[dataset_name]

    validate_required_columns(data,required_columns,dataset_name)

def create_time_stamp(data : pd.DataFrame, dataset_name : str) -> pd.DataFrame :
    data = data.copy()
    date_time_values = (
        data['date'].astype('string').str.strip() + " " + data['time'].astype('string').str.strip()
    )
    data["timestamp"] = pd.to_datetime(
        date_time_values, format="mixed", errors="coerce"
    )
    Invalid_count = data["timestamp"].isna().sum()
    print(f"Timestamp creation : {dataset_name} ")
    print(f"Invalid time stamps : {Invalid_count}")
    return data

def inspect_duplicate_keys(data: pd.DataFrame, dataset_name : str) -> None :
    required_columns = ["road_id", "timestamp"]
    missing_columns = [
        column for column in required_columns 
        if column not in data.columns 
    ] 
    if missing_columns :
        print(
            f"Skipping duplicate-key inspection for "
            f"{dataset_name}: missing columns "
            f"{missing_columns}"
        )
        return
    duplicate_mask = data.duplicated(
        subset = ["road_id", "timestamp"], keep=False
    )
    duplicate_rows = data.loc[duplicate_mask]
    print(f"\nDuplicate-key analysis: {dataset_name}")
    print(
        "Rows involved in duplicate road/timestamp keys:",
        len(duplicate_rows),
    )

    print(
        "Number of duplicate road/timestamp groups:",
        duplicate_rows.groupby(
            ["road_id", "timestamp"]
        ).ngroups,
    )
    if not duplicate_rows.empty:
        print("\nExample duplicate records:")
        print(
            duplicate_rows.sort_values(
                ["road_id", "timestamp"]
            ).head(10)
        )

def inspect_numeric_ranges( df: pd.DataFrame, dataset_name: str)-> None:
    
    print(f"\nNumeric range analysis: {dataset_name}")

    # Columns that should not contain negative values.
    non_negative_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "accident_count",
        "road_capacity",
        "signal_timing",
    ]

    for column in non_negative_columns:
        if column not in df.columns :
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        negative_count = (values < 0).sum()

        print(
            f"{column}: negative values = "
            f"{negative_count}"
        )

    # Check the known speed threshold from the PRD.
    if "avg_speed" in df.columns:
        speed = pd.to_numeric(
            df["avg_speed"],
            errors="coerce",
        )

        high_speed_count = (speed > 200).sum()

        print(
            "avg_speed: values above 200 = "
            f"{high_speed_count}"
        )

    # Check the known occupancy threshold from the PRD.
    if "occupancy" in df.columns:
        occupancy = pd.to_numeric(
            df["occupancy"],
            errors="coerce",
        )

        high_occupancy_count = (occupancy > 100).sum()

        print(
            "occupancy: values above 100 = "
            f"{high_occupancy_count}"
        )

def inspect_missing_values(  df: pd.DataFrame,dataset_name: str)-> None:
   
    print(f"\nMissing-value analysis: {dataset_name}")

    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage.round(2),
    })

    missing_report = missing_report[
        missing_report["missing_count"] > 0
    ]

    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report.sort_values(
            by="missing_count",
            ascending=False,
        ))




def main() -> None :
    csv_files = sorted(RAW_DATA_DIRECTORY.glob("*.csv"))
    if not csv_files :
        raise FileNotFoundError(f"No CSV file found in {RAW_DATA_DIRECTORY}")
    for csv in csv_files :
        data = load_csv(csv.name)
        validate_dataset_schema(data,csv.name)
        inspect_missing_values(data,csv.name)
        data = remove_exact_duplicates(data,csv.name)
        if csv.name in ["traffic_sensor_log.csv","weather_observations.csv"] :
            data = create_time_stamp(data,csv.name)
        if csv.name == "traffic_sensor_log.csv" :
            inspect_duplicate_keys(data,csv.name)
            inspect_numeric_ranges(data,csv.name)
            data = remove_duplicate_traffic_records(data)
        if csv.name ==  "weather_observations.csv" : 
            data = standardize_weather_conditions(data)      
        data = handle_invalid_numeric_values(data,csv.name)
        merged_data = merge_datasets(data)
        inspect_dataframe(csv.name,data)

        
    

if __name__ == "__main__":
    main()
    




















    




    





