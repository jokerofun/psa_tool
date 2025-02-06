from airflow import DAG
from datetime import datetime, timedelta
from src.download_data_operator import DownloadDataOperator
from src.merge_data_operator import MergeDataOperator
from src.preprocess_operator import PreprocessDataOperator
from src.train_model_operator import TrainModelOperator
from src.prediction_operator import PredictionOperator
from src.prescription_operator import PrescriptionOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 2, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('full_pipeline_dag',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    download_gas = DownloadDataOperator(
        task_id='download_gas_data',
        api_endpoint='https://api.example.com/gas',  # Dummy endpoint
        source_name='gas',
        db_path='/tmp/predictions.duckdb',
        if_exists='replace'
    )

    download_elec = DownloadDataOperator(
        task_id='download_electricity_data',
        api_endpoint='https://api.example.com/electricity',  # Dummy endpoint
        source_name='electricity',
        db_path='/tmp/predictions.duckdb',
        if_exists='replace'
    )

    merge_data = MergeDataOperator(
        task_id='merge_historical_data',
        source_info={'gas': 'daily', 'electricity': 'hourly'},
        join_on='timestamp',
        db_path='/tmp/predictions.duckdb',
        merged_table='merged_historical_data'
    )

    preprocess_data = PreprocessDataOperator(
        task_id='preprocess_data',
        input_table='merged_historical_data',
        output_table='preprocessed_data',
        db_path='/tmp/predictions.duckdb'
    )

    train_model = TrainModelOperator(
        task_id='train_model',
        input_table='preprocessed_data',
        model_path='/tmp/model.pkl',
        db_path='/tmp/predictions.duckdb'
    )

    prediction = PredictionOperator(
        task_id='make_prediction',
        prediction_type='weather',  # For demonstration, choose one domain
        data_source='api_placeholder',  # Not used in this simulation
        model_config='{"forecast_horizon": 3, "model": "trained_linear_regression"}',
        db_path='/tmp/predictions.duckdb'
    )

    prescription = PrescriptionOperator(
        task_id='make_prescription',
        prediction_type='weather',
        db_path='/tmp/predictions.duckdb',
        total_quantity=100.0
    )

    # Set up the task dependencies.
    [download_gas, download_elec] >> merge_data
    merge_data >> preprocess_data
    preprocess_data >> train_model
    train_model >> prediction
    prediction >> prescription
