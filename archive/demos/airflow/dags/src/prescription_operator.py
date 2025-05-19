from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import cvxpy as cp
from src import duckdb_utils

class PrescriptionOperator(BaseOperator):
    template_fields = ('prediction_type', 'db_path', 'total_quantity')
    
    @apply_defaults
    def __init__(self, prediction_type: str, db_path: str = 'predictions.duckdb', total_quantity: float = 100.0, *args, **kwargs):
        super(PrescriptionOperator, self).__init__(*args, **kwargs)
        self.prediction_type = prediction_type
        self.db_path = db_path
        self.total_quantity = total_quantity
        
    def execute(self, context):
        self.log.info(f"Running prescription for: {self.prediction_type}")
        conn = duckdb_utils.get_duckdb_connection(db_path=self.db_path, read_only=True)
        result = conn.execute("""
            SELECT forecast_time, predicted_price FROM predictions
            WHERE prediction_type = ?
            ORDER BY forecast_time ASC
        """, (self.prediction_type,)).fetchall()
        conn.close()
        predicted_prices = [row[1] for row in result]
        self.log.info(f"Retrieved predictions: {predicted_prices}")
        
        n = len(predicted_prices)
        x = cp.Variable(n)
        cost = cp.sum(cp.multiply(predicted_prices, x))
        constraints = [x >= 0, cp.sum(x) == self.total_quantity]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        self.log.info(f"Optimized decision variables: {x.value}")
        return x.value
