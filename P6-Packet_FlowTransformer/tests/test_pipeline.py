import unittest
import shutil
import os
import pandas as pd
from pipeline.process import first_pass, second_pass
from pipeline.config import LABEL_MAPPING

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Use a specific test directory instead of tempfile.mkdtemp()
        self.test_dir = 'C:\\Users\\ronie\\OneDrive - Aalborg Universitet\\Documents\\P6\\BenignTrafficTest'

        # Ensure the directory exists
        os.makedirs(self.test_dir, exist_ok=True)

        # Define the output file for the CSV result
        self.output_file = os.path.join(self.test_dir, "output.csv")

        # Use an actual CSV from your dataset
        dataset_dir = 'C:\\Users\\ronie\\OneDrive - Aalborg Universitet\\Documents\\P6'
        actual_csv_file = os.path.join(dataset_dir, "BenignTraffic", "BenignTraffic.csv")

        # Copy the CSV into the test directory
        shutil.copy(actual_csv_file, os.path.join(self.test_dir, "test.csv"))

    def test_pipeline(self):
        first_pass(self.test_dir)
        second_pass(self.test_dir, self.output_file)

        # Check if the output CSV file exists
        self.assertTrue(os.path.exists(self.output_file))

        # Read the output CSV file
        df = pd.read_csv(self.output_file)

        # Print the first two rows of the DataFrame
        print("First two rows of the resulting DataFrame:")
        print(df.head(2))

        # Ensure the 'label' column is present
        self.assertIn("label", df.columns)

        # Ensure the label matches the expected value
        self.assertEqual(df["label"].iloc[0], LABEL_MAPPING["BenignTraffic"])

    def tearDown(self):
        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
