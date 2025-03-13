import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from dag_classes import DataProcessingNode, DataFetchingFromFileNode, DataFetchingFromDBNode, DataFetchingFromAPINode

# filepath: c:\Users\simeo\Desktop\Projects\Git\psa_tool\dag_manager\test_dag_classes.py

class TestDAGNodes(unittest.TestCase):

    def test_data_processing_node(self):
        df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        node = DataProcessingNode("data_node", "sample_data")
        result_df = node.execute(df)
        pd.testing.assert_frame_equal(result_df, df)

    @patch("pandas.read_csv")
    def test_data_fetching_from_file_node(self, mock_read_csv):
        mock_df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        mock_read_csv.return_value = mock_df
        node = DataFetchingFromFileNode("file_node", "dummy_path.csv")
        result_df = node.execute(None)
        pd.testing.assert_frame_equal(result_df, mock_df)
        mock_read_csv.assert_called_once_with("dummy_path.csv")

    @patch("pandas.read_sql")
    def test_data_fetching_from_db_node(self, mock_read_sql):
        mock_df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        mock_read_sql.return_value = mock_df
        node = DataFetchingFromDBNode("db_node", "dummy_table")
        with patch("dag_classes.db_connection", MagicMock()):
            result_df = node.execute(None)
        pd.testing.assert_frame_equal(result_df, mock_df)
        mock_read_sql.assert_called_once_with("SELECT * FROM dummy_table", unittest.mock.ANY)

    @patch("requests.get")
    def test_data_fetching_from_api_node(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [{"column1": 1, "column2": 4}, {"column1": 2, "column2": 5}, {"column1": 3, "column2": 6}]
        mock_get.return_value = mock_response
        node = DataFetchingFromAPINode("api_node", "http://dummy_api.com")
        result_df = node.execute(None)
        expected_df = pd.DataFrame(mock_response.json())
        pd.testing.assert_frame_equal(result_df, expected_df)
        mock_get.assert_called_once_with("http://dummy_api.com")

if __name__ == "__main__":
    unittest.main()