from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from src import duckdb_utils

class PreprocessDataOperator(BaseOperator):
    template_fields = ('input_table', 'output_table', 'db_path')
    
    @apply_defaults
    def __init__(self, input_table: str, output_table: str, db_path: str = 'predictions.duckdb', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_table = input_table
        self.output_table = output_table
        self.db_path = db_path
        
    def execute(self, context):
        df = duckdb_utils.load_dataframe(self.input_table, db_path=self.db_path)
        # Simple preprocessing: sort, fill missing values, and normalize the price column.
        df.sort_values(by='timestamp', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df['price'] = (df['price'] - df['price'].mean()) / df['price'].std()
        duckdb_utils.store_dataframe(df, table_name=self.output_table, db_path=self.db_path, if_exists='replace')
        self.log.info(f"Preprocessed data stored in table {self.output_table}")
        return df.to_json(date_format='iso')
