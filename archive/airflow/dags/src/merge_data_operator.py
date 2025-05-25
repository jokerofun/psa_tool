from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from src import duckdb_utils

class MergeDataOperator(BaseOperator):
    template_fields = ('source_info', 'join_on', 'db_path', 'merged_table')
    
    @apply_defaults
    def __init__(self, source_info: dict, join_on: str, db_path: str = 'predictions.duckdb', merged_table: str = 'merged_historical_data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_info = source_info
        self.join_on = join_on
        self.db_path = db_path
        self.merged_table = merged_table
        
    def execute(self, context):
        merged_df = duckdb_utils.join_and_upscale_historical_data(self.source_info, join_on=self.join_on, db_path=self.db_path)
        duckdb_utils.store_dataframe(merged_df, table_name=self.merged_table, db_path=self.db_path, if_exists='replace')
        self.log.info(f"Merged data stored in table {self.merged_table}")
        return merged_df.to_json(date_format='iso')
