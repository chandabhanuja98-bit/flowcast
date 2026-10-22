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

def main() -> None :
    csv_files = sorted(RAW_DATA_DIRECTORY.glob("*.csv"))
    if not csv_files :
        raise FileNotFoundError(f"No CSV file found in {RAW_DATA_DIRECTORY}")
    for csv in csv_files :
        data = load_csv(csv.name)
        inspect_dataframe(csv.name,data)











if __name__ == "__main__":
    main()




















    




    





