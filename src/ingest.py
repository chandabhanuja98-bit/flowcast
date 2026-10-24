import pandas as pd
from pathlib import Path

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

    


def main() -> None :
    csv_files = sorted(RAW_DATA_DIRECTORY.glob("*.csv"))
    if not csv_files :
        raise FileNotFoundError(f"No CSV file found in {RAW_DATA_DIRECTORY}")
    for csv in csv_files :
        data = load_csv(csv.name)
        if csv.name in ["traffic_sensor_log.csv","weather_observations.csv"] :
            data = create_time_stamp(data,csv.name)
        inspect_dataframe(csv.name,data)
        validate_dataset_schema(data,csv.name)


    

if __name__ == "__main__":
    main()
    




















    




    





