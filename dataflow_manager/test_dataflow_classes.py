import unittest
from unittest.mock import patch
import pandas as pd

from dataflow_classes import DataFetchingFromAPINode, DataFetchingFromFileNode, DataProcessingNode

class TestDataflowNodes(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_data_fetching_from_file(self, mock_read_csv):
        """Test that DataFetchingFromFileNode correctly fetches data from a file."""
        # Mocking pandas read_csv to return a DataFrame
        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        node = DataFetchingFromFileNode(name="pricesEUR", source="fake.csv")
        node.run()
        results = node.get_results()

        self.assertIn("pricesEUR", results)
        self.assertIsInstance(results["pricesEUR"], pd.DataFrame)
        self.assertEqual(len(results["pricesEUR"]), 2)  # Expecting 2 rows

    @patch("requests.get")
    def test_data_fetching_from_api(self, mock_requests_get):
        """Test that DataFetchingFromAPINode correctly fetches data from an API."""
        mock_requests_get.return_value.json.return_value = [{"price": 10}, {"price": 20}]

        node = DataFetchingFromAPINode(name="apiPrices", source="http://fakeapi.com/data")
        node.run()
        results = node.get_results()

        self.assertIn("apiPrices", results)
        self.assertIsInstance(results["apiPrices"], pd.DataFrame)
        self.assertEqual(len(results["apiPrices"]), 2)  # Expecting 2 rows

    def test_data_processing_node(self):
        """Test that DataProcessingNode correctly modifies data."""
        def sample_processing(dfs):
            dfs["processed"] = dfs["input"] * 2  # Simple transformation
            return dfs

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

            file_node = DataFetchingFromFileNode(name="input", source="fake.csv")
            process_node = DataProcessingNode(name="processor", process_func=sample_processing)

            file_node >> process_node  # Set dependency

            process_node.run()

            results = process_node.get_results()

            # Assertions
            self.assertIn("processed", results)
            self.assertIsInstance(results["processed"], pd.DataFrame)
            self.assertTrue((results["processed"]["col1"] == [2, 4]).all())  # Check transformation

    @patch.object(DataFetchingFromFileNode, "run")
    def test_dependency_execution_order(self, mock_run):
        """Test that dependencies are executed before processing nodes."""
        mock_run.side_effect = lambda: None  # Mock to prevent actual execution

        pricesEUR = DataFetchingFromFileNode(name="pricesEUR", source="fake.csv")
        processNode = DataProcessingNode(name="process", process_func=lambda x: x)

        pricesEUR >> processNode  # Setting dependency

        with patch.object(processNode, "process", wraps=processNode.process) as mock_process:
            processNode.run()
            mock_run.assert_called()
            mock_process.assert_called()

if __name__ == "__main__":
    unittest.main()
