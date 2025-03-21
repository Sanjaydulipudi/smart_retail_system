import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine
from config import get_config

class DataProcessor:
    """Class for processing retail sales data."""
    
    def __init__(self):
        self.config = get_config()
        self.raw_data_path = self.config.DATA_PATH
        self.processed_data_path = self.config.PROCESSED_DATA_PATH
        
    def generate_sample_data(self, days=365, start_date='2023-01-01'):
        """Generate sample sales data for testing."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        dates = [start + timedelta(days=i) for i in range(days)]
        
        # Create base sales with weekly seasonality
        base_sales = np.random.normal(1500, 300, days)
        
        # Add weekly pattern (weekends have higher sales)
        weekly_pattern = np.array([1.0, 1.0, 1.0, 1.1, 1.2, 1.5, 1.3] * (days//7 + 1))[:days]
        
        # Add monthly seasonality
        monthly_pattern = np.array([1.0, 0.9, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.2, 1.5, 1.7, 1.3] * (days//365 + 1))
        monthly_indices = [(start + timedelta(days=i)).month - 1 for i in range(days)]
        monthly_factors = np.array([monthly_pattern[i] for i in monthly_indices])
        
        # Add trend
        trend = np.linspace(1.0, 1.2, days)
        
        # Combine all factors
        sales = base_sales * weekly_pattern * monthly_factors * trend
        
        # Create data dictionary
        data = {
            'date': [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)],
            'day': [(start + timedelta(days=i)).day for i in range(days)],
            'month': [(start + timedelta(days=i)).month for i in range(days)],
            'year': [(start + timedelta(days=i)).year for i in range(days)],
            'weekday': [(start + timedelta(days=i)).weekday() for i in range(days)],
            'sales': sales.astype(int),
            'items_sold': (sales / 25).astype(int),  # Average item price of $25
            'transactions': (sales / 75).astype(int),  # Average transaction of $75
            'customer_count': (sales / 100).astype(int)  # Average spend per customer of $100
        }
        
        # Create product categories with different sales distributions
        categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Beauty']
        category_percentages = {
            'Electronics': np.random.normal(0.3, 0.05, days),
            'Clothing': np.random.normal(0.25, 0.05, days),
            'Home': np.random.normal(0.2, 0.03, days),
            'Food': np.random.normal(0.15, 0.03, days),
            'Beauty': np.random.normal(0.1, 0.02, days)
        }
        
        # Normalize percentages to sum to 1
        for i in range(days):
            total = sum(category_percentages[cat][i] for cat in categories)
            for cat in categories:
                category_percentages[cat][i] /= total
                
        # Add category sales to data
        for cat in categories:
            data[f'{cat.lower()}_sales'] = (sales * category_percentages[cat]).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        df.to_csv(self.raw_data_path, index=False)
        
        return df
    
    def load_data(self):
        """Load data from CSV file or generate if not exists."""
        if not os.path.exists(self.raw_data_path):
            return self.generate_sample_data()
        return pd.read_csv(self.raw_data_path)
    
    def process_data(self):
        """Process the raw data for model training."""
        df = self.load_data()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Feature engineering
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month_progress'] = df['day'] / df.groupby('month')['day'].transform('max')
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate moving averages
        df['sales_ma7'] = df['sales'].rolling(window=7, min_periods=1).mean()
        df['sales_ma30'] = df['sales'].rolling(window=30, min_periods=1).mean()
        
        # Calculate sales growth and fill NaN values
        df['sales_growth'] = df['sales'].pct_change(periods=7)
        df['sales_growth'].fillna(0, inplace=True)  # Fill NaN values with 0
        
        # Add holidays (simplified version - could be expanded)
        holidays = ['2023-01-01', '2023-12-25', '2023-07-04', '2023-11-24']
        df['is_holiday'] = df['date'].isin(pd.to_datetime(holidays)).astype(int)
        
        # Save processed data
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        df.to_csv(self.processed_data_path, index=False)
        
        return df
    
    def export_to_db(self, df=None, table_name='sales_data'):
        """Export data to SQL database."""
        if df is None:
            df = self.process_data()
            
        engine = create_engine(self.config.SQLALCHEMY_DATABASE_URI)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        return True


if __name__ == "__main__":
    processor = DataProcessor()
    processed_data = processor.process_data()
    processor.export_to_db(processed_data)
    print(f"Data processed and exported. Shape: {processed_data.shape}")