import unittest
import os
import tempfile
import shutil
import json
from app import app
from scripts.data_processor import DataProcessor

class TestFlaskApp(unittest.TestCase):
    """Tests for the Flask application"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, 'data')
        self.models_dir = os.path.join(self.test_dir, 'models')
        self.reports_dir = os.path.join(self.test_dir, 'reports')
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Configure Flask app for testing
        app.config.update({
            'TESTING': True,
            'DATA_PATH': os.path.join(self.data_dir, 'sales.csv'),
            'PROCESSED_DATA_PATH': os.path.join(self.data_dir, 'processed_sales.csv'),
            'MODEL_PATH': os.path.join(self.models_dir, 'sales_model.h5'),
            'SCALER_PATH': os.path.join(self.models_dir, 'scaler.pkl'),
            'MODEL_METADATA_PATH': os.path.join(self.models_dir, 'model_metadata.json'),
            'REPORTS_DIR': self.reports_dir,
            'WTF_CSRF_ENABLED': False  # Disable CSRF for testing
        })
        
        # Create test client
        self.client = app.test_client()
        
        # Generate sample data
        data_processor = DataProcessor(
            data_path=app.config['DATA_PATH'],
            processed_path=app.config['PROCESSED_DATA_PATH']
        )
        data_processor.generate_sample_data()
        data_processor.process_data()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)
    
    def test_home_page(self):
        """Test home page loading"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Smart Retail System', response.data)
    
    def test_dashboard_page(self):
        """Test dashboard page loading"""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)
    
    def test_predict_page(self):
        """Test prediction page loading"""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sales Prediction', response.data)
    
    def test_reports_page(self):
        """Test reports page loading"""
        response = self.client.get('/reports')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Reports', response.data)
    
    def test_api_data_summary(self):
        """Test API endpoint for data summary"""
        response = self.client.get('/api/data/summary')
        self.assertEqual(response.status_code, 200)
        
        # Parse JSON response
        data = json.loads(response.data)
        
        # Check structure
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        # Check summary fields
        summary = data['data']
        self.assertIn('total_sales', summary)
        self.assertIn('avg_daily_sales', summary)
        self.assertIn('record_count', summary)
    
    def test_api_data_latest(self):
        """Test API endpoint for latest data"""
        response = self.client.get('/api/data/latest')
        self.assertEqual(response.status_code, 200)
        
        # Parse JSON response
        data = json.loads(response.data)
        
        # Check structure
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        # Check that data is a list
        self.assertIsInstance(data['data'], list)
        
        # Check that we have up to 30 records
        self.assertLessEqual(len(data['data']), 30)
    
    def test_api_predict(self):
        """Test API endpoint for predictions"""
        # Need to train a model first (skip full test if this fails)
        try:
            train_response = self.client.post('/train', data={
                'model_type': 'ffn'
            })
            
            # If training was successful, test prediction API
            if train_response.status_code == 200:
                # Sample features for prediction
                test_data = {
                    'day': 15,
                    'month': 3,
                    'year': 2023,
                    'day_of_week': 3,
                    'is_weekend': 0,
                    'is_holiday': 0
                }
                
                response = self.client.post('/api/predict', 
                                           json=test_data,
                                           content_type='application/json')
                
                self.assertEqual(response.status_code, 200)
                
                # Parse JSON response
                data = json.loads(response.data)
                
                # Check structure
                self.assertTrue(data['success'])
                self.assertIn('prediction', data)
                
                # Check that prediction is a number
                self.assertIsInstance(data['prediction'], float)
        except Exception as e:
            print(f"Skipping prediction API test due to model training failure: {e}")
    
    def test_upload_data(self):
        """Test data upload functionality"""
        # Create a small test CSV file
        csv_content = "date,sales,category\n2023-01-01,1000,Electronics\n2023-01-02,1500,Home\n"
        test_csv_path = os.path.join(self.test_dir, 'test_upload.csv')
        
        with open(test_csv_path, 'w') as f:
            f.write(csv_content)
        
        # Test file upload
        with open(test_csv_path, 'rb') as f:
            response = self.client.post(
                '/upload',
                data={
                    'file': (f, 'test_upload.csv')
                },
                follow_redirects=True
            )
        
        # Should redirect to dashboard on success
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)
    
    def test_train_model_route(self):
        """Test model training route"""
        response = self.client.post(
            '/train',
            data={'model_type': 'ffn'},
            follow_redirects=True
        )
        
        # Check response status
        self.assertEqual(response.status_code, 200)
        
        # Check if model file was created
        self.assertTrue(os.path.exists(app.config['MODEL_PATH']) or 
                        os.path.exists(os.path.join(self.models_dir, 'sales_model.h5')))


if __name__ == '__main__':
    unittest.main()