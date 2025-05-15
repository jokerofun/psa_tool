from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from src import duckdb_utils

class TrainModelOperator(BaseOperator):
    template_fields = ('input_table', 'model_path', 'db_path')
    
    @apply_defaults
    def __init__(self, input_table: str, model_path: str, db_path: str = 'predictions.duckdb', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_table = input_table
        self.model_path = model_path
        self.db_path = db_path
        
    def execute(self, context):
        df = duckdb_utils.load_dataframe(self.input_table, db_path=self.db_path)
        # For demonstration, create a dummy target variable.
        df['target'] = df['price'].shift(-1)
        df.dropna(inplace=True)
        X = df[['price']]
        y = df['target']
        model = LinearRegression()
        model.fit(X, y)
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        self.log.info(f"Model trained and saved to {self.model_path}")
        return self.model_path
