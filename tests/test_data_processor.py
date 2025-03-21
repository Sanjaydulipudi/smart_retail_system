import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from scripts.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Tests for the DataProcessor class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create instance with test directory
        self.data_processor = DataProcessor(
            data_path=os.path.join(self.data_dir, 'sales.csv'),
            processed_path=os.path.join(self.data_dir, 'processed_sales.csv')
        )
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        self.data_processor.generate_sample_data(rows=100)
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.data_processor.data_path))
        
        # Load and verify data
        df = pd.read_csv(self.data_processor.data_path)
        self.assertEqual(len(df), 100)
        self.assertTrue('date' in df.columns)
        self.assertTrue('sales' in df.columns)
        self.assertTrue('category' in df.columns)
    
    def test_process_data(self):
        """Test data processing"""
        # Generate sample data first
        self.data_processor.generate_sample_data()
        
        # Process the data
        self.data_processor.process_data()
        
        # Check that processed file was created
        self.assertTrue(os.path.exists(self.data_processor.processed_path))
        
        # Load and verify processed data
        df = pd.read_csv(self.data_processor.processed_path)
        
        # Check that date features were created
        self.assertTrue('day' in df.columns)
        self.assertTrue('month' in df.columns)
        self.assertTrue('year' in df.columns)
        self.assertTrue('day_of_week' in df.columns)
        
        # Check that there are no missing values
        self.assertEqual(df.isnull().sum().sum(), 0)
    
    def test_load_data(self):
        """Test data loading"""
        # Generate and process sample data
        self.data_processor.generate_sample_data()
        self.data_processor.process_data()
        
        # Load the data
        df = self.data_processor.load_data()
        
        # Check that it returns a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check that it has the expected columns
        required_columns = ['date', 'sales', 'category', 'day', 'month', 'year']
        for col in required_columns:
            self.assertTrue(col in df.columns)
    
    def test_get_features_targets(self):
        """Test extracting features and targets"""
        # Generate and process sample data
        self.data_processor.generate_sample_data()
        self.data_processor.process_data()
        
        # Get features and targets
        X, y = self.data_processor.get_features_targets()
        
        # Check types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        
        # Check shapes (features should have multiple columns, target just one)
        self.assertTrue(X.shape[0] > 0)  # At least one row
        self.assertTrue(X.shape[1] > 1)  # Multiple feature columns
        self.assertEqual(y.shape[1], 1)  # One target column


if __name__ == '__main__':
    unittest.main()