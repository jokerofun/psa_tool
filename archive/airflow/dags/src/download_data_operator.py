from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import pandas as pd
from src import duckdb_utils

class DownloadDataOperator(BaseOperator):
    template_fields = ('api_endpoint', 'source_name', 'db_path', 'if_exists')
    
    @apply_defaults
    def __init__(self, api_endpoint: str, source_name: str, db_path: str = 'predictions.duckdb', if_exists: str = 'replace', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_endpoint = api_endpoint
        self.source_name = source_name
        self.db_path = db_path
        self.if_exists = if_exists
        
    def execute(self, context):
        # Simulate an API download. In a real case, use requests or another HTTP client.
        if self.source_name.lower() == 'gas':
            df = pd.DataFrame({
                'timestamp': pd.date_range(start='2025-02-01', periods=3, freq='D'),
                'price': [2.5, 2.7, 2.6]
            })
        elif self.source_name.lower() == 'electricity':
            df = pd.DataFrame({
                'timestamp': pd.date_range(start='2025-02-01', periods=48, freq='H'),
                'price': [50 + i*0.1 for i in range(48)]
            })
        else:
            raise ValueError("Unsupported source")
        
        duckdb_utils.store_historical_data(df, source_name=self.source_name, db_path=self.db_path, if_exists=self.if_exists)
        self.log.info(f"Downloaded and stored data for {self.source_name}")
        return df.to_json(date_format='iso')
