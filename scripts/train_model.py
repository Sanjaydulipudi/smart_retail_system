import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
import json
import logging
from config import get_config



# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesPredictionModel:
    """Class for training and evaluating sales prediction models."""
    
    def __init__(self):
        self.config = get_config()
        self.processed_data_path = self.config.PROCESSED_DATA_PATH
        self.model_path = self.config.MODEL_PATH
        self.scaler = StandardScaler()
        self.model = None
        self.lookback = 30  # Number of days to look back for time series models
        self.metrics = {}
        
    def prepare_features(self, df):
        """Prepare features for model training."""
        # Select features
        feature_columns = [
            'day', 'month', 'year', 'day_of_week', 
            'day_of_year', 'is_weekend', 'is_holiday'
        ]
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=['month', 'day_of_week'])
        
        # Add lag features for time series models
        for lag in [1, 7, 14, 30]:
            df_encoded[f'sales_lag_{lag}'] = df_encoded['sales'].shift(lag)
        
        # Fill NaN values before dropping rows
        # This handles the NaN values in sales_growth and other calculated fields
        if 'sales_growth' in df_encoded.columns:
            df_encoded['sales_growth'].fillna(0, inplace=True)
        
        # Fill NaN values in lag features
        for lag in [1, 7, 14, 30]:
            df_encoded[f'sales_lag_{lag}'].fillna(df_encoded['sales'].mean(), inplace=True)
        
        # Drop any remaining NaN values
        df_encoded = df_encoded.dropna()
        
        # Prepare X and y
        X = df_encoded.drop(['date', 'sales'], axis=1)
        y = df_encoded['sales']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Save the scaler
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.scaler, os.path.join(os.path.dirname(self.model_path), 'scaler.pkl'))
        
        return X_scaled, y, list(X.columns)
    
    def build_linear_model(self):
        """Build a linear regression model."""
        return LinearRegression()
    
    def build_random_forest_model(self):
        """Build a random forest regression model."""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
    
    def prepare_time_series_data(self, X, y):
        """Prepare data for time series model."""
        # Create a dataset with lookback window
        X_ts = []
        y_ts = []
        
        for i in range(len(X) - self.lookback):
            X_ts.append(X[i:i+self.lookback].flatten())  # Flatten the window
            y_ts.append(y.iloc[i+self.lookback])
            
        return np.array(X_ts), np.array(y_ts)
    
    def train_model(self, model_type='linear'):
        """Train the selected model type."""
        # Load processed data
        df = pd.read_csv(self.processed_data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        if model_type == 'linear':
            # Build and train linear model
            self.model = self.build_linear_model()
            self.model.fit(X_train, y_train)
            
            # Get predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics for history
            history['train_loss'] = [mean_squared_error(y_train, y_train_pred)]
            history['val_loss'] = [mean_squared_error(y_test, y_test_pred)]
            history['train_mae'] = [mean_absolute_error(y_train, y_train_pred)]
            history['val_mae'] = [mean_absolute_error(y_test, y_test_pred)]
            print('model trainined successful')
            
        elif model_type == 'rf':
            # Build and train random forest
            self.model = self.build_random_forest_model()
            self.model.fit(X_train, y_train)
            
            # Get predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics for history
            history['train_loss'] = [mean_squared_error(y_train, y_train_pred)]
            history['val_loss'] = [mean_squared_error(y_test, y_test_pred)]
            history['train_mae'] = [mean_absolute_error(y_train, y_train_pred)]
            history['val_mae'] = [mean_absolute_error(y_test, y_test_pred)]
            
        elif model_type == 'time_series':
            # Prepare time series data
            X_ts_train, y_ts_train = self.prepare_time_series_data(X_train, y_train.reset_index(drop=True))
            X_ts_test, y_ts_test = self.prepare_time_series_data(X_test, y_test.reset_index(drop=True))
            
            # Build and train random forest for time series
            self.model = self.build_random_forest_model()
            self.model.fit(X_ts_train, y_ts_train)
            
            # Get predictions
            y_train_pred = self.model.predict(X_ts_train)
            y_test_pred = self.model.predict(X_ts_test)
            
            # Calculate metrics for history
            history['train_loss'] = [mean_squared_error(y_ts_train, y_train_pred)]
            history['val_loss'] = [mean_squared_error(y_ts_test, y_test_pred)]
            history['train_mae'] = [mean_absolute_error(y_ts_train, y_train_pred)]
            history['val_mae'] = [mean_absolute_error(y_ts_test, y_test_pred)]
            
            # Update test data for final evaluation
            X_test, y_test = X_ts_test, y_ts_test
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Save the model
        joblib.dump(self.model, self.model_path)
            
        # Final evaluation
        if model_type == 'time_series':
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
        else:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
        # Store metrics
        self.metrics = {
            'model_type': model_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'feature_count': X_train.shape[1]
        }
        
        # For random forest, also store feature importance
        if model_type in ['rf', 'time_series']:
            feature_importance = {}
            if model_type == 'rf':
                for i, importance in enumerate(self.model.feature_importances_):
                    feature_importance[feature_names[i]] = float(importance)
            else:
                # For time series data, we have different feature structure
                feature_importance = {'time_window': 1.0}
                
            self.metrics['feature_importance'] = feature_importance
        
        # Save metrics
        # with open(os.path.join(os.path.dirname(self.model_path), 'model_metadata.json'), 'w') as f:
        #     json.dump(self.metrics, f, indent=4)
            
        # Plot training history
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(history['train_loss'], label='Training Loss')
        # plt.plot(history['val_loss'], label='Validation Loss')
        # plt.title('Model Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss (MSE)')
        # plt.legend()
        
        # plt.subplot(1, 2, 2)
        # plt.plot(history['train_mae'], label='Training MAE')
        # plt.plot(history['val_mae'], label='Validation MAE')
        # plt.title('Model Mean Absolute Error')
        # plt.xlabel('Epoch')
        # plt.ylabel('MAE')
        # plt.legend()
        
        # Save the plot
        # os.makedirs(self.config.REPORTS_DIR, exist_ok=True)
        # plt.savefig(os.path.join(self.config.REPORTS_DIR, 'model_training_history.png'))
        
        # # Plot feature importance if available
        # if model_type in ['rf', 'time_series'] and model_type != 'time_series':
        #     plt.figure(figsize=(10, 8))
        #     features = np.array(feature_names)
        #     importances = self.model.feature_importances_
        #     indices = np.argsort(importances)[::-1]
            
        #     plt.title('Feature Importance')
        #     plt.bar(range(len(indices)), importances[indices], color='b', align='center')
        #     plt.xticks(range(len(indices)), features[indices], rotation=90)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(self.config.REPORTS_DIR, f'feature_importance_{model_type}.png'))
        
        # logger.info(f"Model training completed. Type: {model_type}, MAE: {self.metrics['mae']:.2f}")
        return self.metrics
    
    def predict(self, features):
        """Make predictions using the trained model."""
        if self.model is None:
            self.model = joblib.load(self.model_path)
            
        # Load scaler if not initialized
        if not hasattr(self, 'scaler') or self.scaler is None:
            scaler_path = os.path.join(os.path.dirname(self.model_path), 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Make prediction
        return self.model.predict(X_scaled)
    

if __name__ == "__main__":
    model = SalesPredictionModel()
    metrics = model.train_model(model_type='linear')
    print(f"Model trained successfully. MAE: {metrics['mae']:.2f}")