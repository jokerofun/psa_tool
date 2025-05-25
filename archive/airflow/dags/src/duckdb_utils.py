import duckdb
import pandas as pd

default_db_path = '/data/duckdb/predictions.duckdb'

def get_duckdb_connection(db_path: str = default_db_path, read_only: bool = False):
    """
    Returns a DuckDB connection.
    """
    return duckdb.connect(database=db_path, read_only=read_only)

def store_dataframe(df: pd.DataFrame, table_name: str, db_path: str = default_db_path, if_exists: str = 'replace'):
    """
    Stores a Pandas DataFrame into a DuckDB table.
    """
    conn = get_duckdb_connection(db_path=db_path, read_only=False)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.commit()
    conn.close()

def load_dataframe(table_name: str, db_path: str = default_db_path) -> pd.DataFrame:
    """
    Loads a DuckDB table into a Pandas DataFrame.
    """
    conn = get_duckdb_connection(db_path=db_path, read_only=True)
    df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
    conn.close()
    return df

def store_historical_data(df: pd.DataFrame, source_name: str, db_path: str = default_db_path, if_exists: str = 'append'):
    """
    Stores historical data for a given source in a table named 'historical_{source_name}'.
    """
    table_name = f"historical_{source_name}"
    store_dataframe(df, table_name, db_path=db_path, if_exists=if_exists)

def load_historical_data(source_name: str, db_path: str = default_db_path) -> pd.DataFrame:
    """
    Loads historical data for a given source from table 'historical_{source_name}'.
    """
    table_name = f"historical_{source_name}"
    return load_dataframe(table_name, db_path=db_path)

def upscale_daily_to_hourly(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Upscales daily data to hourly data by repeating each row 24 times with updated timestamps.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    upscaled_rows = []
    for _, row in df.iterrows():
        base_time = row[datetime_col].normalize()  # get midnight of the day
        hourly_times = pd.date_range(start=base_time, periods=24, freq='H')
        temp_df = pd.DataFrame([row] * len(hourly_times))
        temp_df[datetime_col] = hourly_times
        upscaled_rows.append(temp_df)
    upscaled_df = pd.concat(upscaled_rows, ignore_index=True)
    return upscaled_df

def join_and_upscale_historical_data(source_info: dict, join_on: str, db_path: str = default_db_path) -> pd.DataFrame:
    """
    Joins historical data tables from different sources. For sources with lower frequency
    (e.g. daily) it will upscale to the highest frequency (e.g. hourly) by repeating rows.
    
    :param source_info: A dict mapping source names to frequency strings ('daily' or 'hourly').
    :param join_on: The datetime column to join on.
    """
    if not source_info:
        raise ValueError("At least one source must be provided.")
    
    # Determine the highest frequency: assume hourly > daily.
    highest_freq = 'hourly' if any(freq.lower() == 'hourly' for freq in source_info.values()) else 'daily'
    
    dataframes = {}
    for source, freq in source_info.items():
        df = load_historical_data(source, db_path=db_path)
        df[join_on] = pd.to_datetime(df[join_on])
        if freq.lower() == 'daily' and highest_freq == 'hourly':
            df = upscale_daily_to_hourly(df, join_on)
        dataframes[source] = df

    sources = list(dataframes.keys())
    joined_df = dataframes[sources[0]]
    for source in sources[1:]:
        joined_df = pd.merge(joined_df, dataframes[source], on=join_on, how='outer', suffixes=('', f'_{source}'))
    
    joined_df.sort_values(by=join_on, inplace=True)
    return joined_df
