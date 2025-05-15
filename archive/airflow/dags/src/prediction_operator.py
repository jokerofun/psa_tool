from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import json
from datetime import datetime
from src import duckdb_utils

class PredictionOperator(BaseOperator):
    # Templated fields to allow dynamic configuration.
    template_fields = ('prediction_type', 'data_source', 'model_config', 'db_path')
    
    @apply_defaults
    def __init__(self, prediction_type: str, data_source: str, model_config: str, db_path: str = 'predictions.duckdb', *args, **kwargs):
        """
        :param prediction_type: E.g. 'stock' or 'weather'
        :param data_source: A (templated) data source identifier (for demo purposes)
        :param model_config: A JSON string with model parameters
        :param db_path: Path to the DuckDB file.
        """
        super(PredictionOperator, self).__init__(*args, **kwargs)
        self.prediction_type = prediction_type
        self.data_source = data_source
        self.model_config = model_config
        self.db_path = db_path

    def execute(self, context):
        self.log.info(f"Running prediction for: {self.prediction_type}")
        self.log.info(f"Data source: {self.data_source}")
        self.log.info(f"Model config: {self.model_config}")
        config = json.loads(self.model_config)
        
        # Simulate prediction logic (here simply create a dummy forecast).
        now = datetime.now()
        predictions = {
            now.replace(hour=0, minute=0, second=0, microsecond=0): 50.0,
            now.replace(hour=1, minute=0, second=0, microsecond=0): 52.0,
            now.replace(hour=2, minute=0, second=0, microsecond=0): 48.0,
        }
        if self.prediction_type.lower() == 'stock':
            predictions = {k: v + 10 for k, v in predictions.items()}
        elif self.prediction_type.lower() == 'weather':
            predictions = {k: v - 5 for k, v in predictions.items()}
        else:
            raise ValueError("Unsupported prediction type")
        
        # Write predictions into DuckDB (a table called 'predictions' is used).
        conn = duckdb_utils.get_duckdb_connection(db_path=self.db_path, read_only=False)
        # Create table if needed.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                forecast_time TIMESTAMP,
                predicted_price DOUBLE,
                prediction_type VARCHAR
            )
        """)
        conn.execute("DELETE FROM predictions WHERE prediction_type = ?", (self.prediction_type,))
        for forecast_time, predicted_price in predictions.items():
            conn.execute("INSERT INTO predictions VALUES (?, ?, ?)", (forecast_time, predicted_price, self.prediction_type))
        conn.commit()
        conn.close()
        self.log.info(f"Predictions for {self.prediction_type} stored: {predictions}")
        return predictions
