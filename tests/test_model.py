import unittest
import os
import numpy as np
import tempfile
import shutil
import tensorflow as tf
import pickle
import json
from scripts.train_model import SalesPredictionModel
from scripts.data_processor import DataProcessor

class TestSalesPredictionModel(unittest.TestCase):
    """Tests for the SalesPredictionModel class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, 'data')
        self.models_dir = os.path.join(self.test_dir, 'models')
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create DataProcessor with test directory
        self.data_processor = DataProcessor(
            data_path=os.path.join(self.data_dir, 'sales.csv'),
            processed_path=os.path.join(self.data_dir, 'processed_sales.csv')
        )
        
        # Generate and process data
        self.data_processor.generate_sample_data(rows=200)
        self.data_processor.process_data()
        
        # Create model instance
        self.model = SalesPredictionModel(
            model_path=os.path.join(self.models_dir, 'sales_model.h5'),
            scaler_path=os.path.join(self.models_dir, 'scaler.pkl'),
            metadata_path=os.path.join(self.models_dir, 'model_metadata.json')
        )
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)
    
    def test_build_ffn_model(self):
        """Test building feedforward neural network"""
        # Get feature dimensions
        X, _ = self.data_processor.get_features_targets()
        input_dim = X.shape[1]
        
        # Build model
        model = self.model._build_ffn_model(input_dim)
        
        # Check that it's a valid TensorFlow model
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape[1], input_dim)
        
        # Check output shape (should be 1 for regression)
        self.assertEqual(model.output_shape[1], 1)
    
    def test_build_lstm_model(self):
        """Test building LSTM model"""
        # Build model with sample parameters
        model = self.model._build_lstm_model(
            sequence_length=7,
            features=5,
            lstm_units=32
        )
        
        # Check that it's a valid TensorFlow model
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape (batch_size, sequence_length, features)
        self.assertEqual(model.input_shape[1:], (7, 5))
        
        # Check output shape (should be 1 for regression)
        self.assertEqual(model.output_shape[1], 1)
    
    def test_train_model(self):
        """Test model training"""
        # Train a simple model with minimal epochs
        metrics = self.model.train_model(
            model_type='ffn',
            epochs=2,
            batch_size=32,
            data_processor=self.data_processor
        )
        
        # Check that metrics were returned
        self.assertIsInstance(metrics, dict)
        self.assertTrue('train_loss' in metrics)
        self.assertTrue('val_loss' in metrics)
        
        # Check that model file was created
        self.assertTrue(os.path.exists(self.model.model_path))
        
        # Check that scaler was saved
        self.assertTrue(os.path.exists(self.model.scaler_path))
        
        # Check that metadata was saved
        self.assertTrue(os.path.exists(self.model.metadata_path))
        
        # Load and check metadata
        with open(self.model.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertTrue('model_type' in metadata)
        self.assertEqual(metadata['model_type'], 'ffn')
    
    def test_predict(self):
        """Test making predictions"""
        # Train a simple model first
        self.model.train_model(
            model_type='ffn',
            epochs=2,
            batch_size=32,
            data_processor=self.data_processor
        )
        
        # Get some sample data
        X, _ = self.data_processor.get_features_targets()
        sample_data = X[:5]
        
        # Make predictions
        predictions = self.model.predict(sample_data)
        
        # Check prediction shape and type
        self.assertEqual(predictions.shape, (5, 1))
        self.assertIsInstance(predictions, np.ndarray)
        
        # Check that predictions are reasonable sales values (positive)
        self.assertTrue(np.all(predictions >= 0))


if __name__ == '__main__':
    unittest.main()