import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.dates as mdates
from config import get_config
from scripts.data_processor import DataProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class for generating analytical reports and visualizations."""
    
    def __init__(self):
        self.config = get_config()
        self.reports_dir = self.config.REPORTS_DIR
        self.data_path = self.config.PROCESSED_DATA_PATH
        self.model_path = self.config.MODEL_PATH
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_data(self):
        """Load the processed sales data."""
        if not os.path.exists(self.data_path):
            # Generate data if it doesn't exist
            processor = DataProcessor()
            df = processor.process_data()
        else:
            df = pd.read_csv(self.data_path)
            df['date'] = pd.to_datetime(df['date'])
            
        return df
    
    def generate_sales_trend_report(self):
        """Generate sales trend visualization."""
        df = self.load_data()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['sales'], label='Daily Sales')
        
        # Add moving averages
        if 'sales_ma7' in df.columns:
            plt.plot(df['date'], df['sales_ma7'], label='7-Day Moving Average', 
                     linestyle='--', color='orange')
        if 'sales_ma30' in df.columns:
            plt.plot(df['date'], df['sales_ma30'], label='30-Day Moving Average', 
                     linestyle='--', color='red')
        
        plt.title('Sales Trend Analysis')
        plt.xlabel('Date')
        plt.ylabel('Sales (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.reports_dir, 'sales_trend.png')
        plt.savefig(output_path, dpi=300)
        logger.info(f"Sales trend report generated: {output_path}")
        
        return output_path
    
    def generate_category_analysis(self):
        """Generate category-wise sales analysis."""
        df = self.load_data()
        
        # Identify category columns
        category_cols = [col for col in df.columns if col.endswith('_sales')]
        
        if not category_cols:
            logger.warning("No category sales columns found in data")
            return None
        
        # Create a new dataframe with date and categories
        cat_df = df[['date'] + category_cols].copy()
        
        # Monthly aggregation
        monthly_df = cat_df.set_index('date').resample('M').sum().reset_index()
        monthly_df['month'] = monthly_df['date'].dt.strftime('%Y-%m')
        
        # Create stacked bar chart
        plt.figure(figsize=(14, 7))
        
        # Plot stacked bars
        bottom = np.zeros(len(monthly_df))
        for col in category_cols:
            category_name = col.replace('_sales', '').title()
            plt.bar(monthly_df['month'], monthly_df[col], bottom=bottom, label=category_name)
            bottom += monthly_df[col].values
        
        plt.title('Monthly Sales by Category')
        plt.xlabel('Month')
        plt.ylabel('Sales (USD)')
        plt.legend(title='Categories')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.reports_dir, 'category_analysis.png')
        plt.savefig(output_path, dpi=300)
        logger.info(f"Category analysis report generated: {output_path}")
        
        # Create pie chart of total sales by category
        plt.figure(figsize=(10, 8))
        total_by_category = cat_df[category_cols].sum()
        labels = [col.replace('_sales', '').title() for col in category_cols]
        plt.pie(total_by_category, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Total Sales Distribution by Category')
        plt.tight_layout()
        
        # Save the pie chart
        pie_output_path = os.path.join(self.reports_dir, 'category_distribution.png')
        plt.savefig(pie_output_path, dpi=300)
        
        return output_path
    
    def generate_weekly_pattern_report(self):
        """Generate weekly sales pattern visualization."""
        df = self.load_data()
        
        # Ensure day_of_week exists
        if 'day_of_week' not in df.columns and 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
        
        # Create weekday names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Group by day of week
        weekly_pattern = df.groupby('day_of_week')['sales'].agg(['mean', 'std']).reset_index()
        weekly_pattern['day_name'] = weekly_pattern['day_of_week'].apply(lambda x: day_names[x])
        
        # Create bar chart with error bars
        plt.figure(figsize=(10, 6))
        plt.bar(weekly_pattern['day_name'], weekly_pattern['mean'], 
                yerr=weekly_pattern['std'], capsize=5, alpha=0.7)
        plt.title('Average Sales by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Sales (USD)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.reports_dir, 'weekly_pattern.png')
        plt.savefig(output_path, dpi=300)
        logger.info(f"Weekly pattern report generated: {output_path}")
        
        return output_path
    
    def generate_sales_forecast(self, days=30, model_type='ffn', confidence_interval=0.95):
        """
        Generate sales forecast visualization.
        
        Parameters:
        days (int): Number of days to forecast
        model_type (str): Type of model to use ('ffn' or 'lstm')
        confidence_interval (float): Confidence interval for prediction (0.0-1.0)
        
        Returns:
        str: Path to the generated forecast image
        """
        df = self.load_data()
        
        # Load model based on model_type
        model_file = f"{model_type}_model.h5"
        model_path = os.path.join(os.path.dirname(self.model_path), model_file)
        
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded {model_type} model from {model_path}")
        except (OSError, IOError) as e:
            logger.error(f"Could not load model {model_type}: {e}")
            logger.info("Using fallback prediction method")
            model = None
        
        # Get latest data for prediction
        latest_data = df.tail(30).copy()
        
        # Generate future dates
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Make forecast using model if available, otherwise use placeholder
        if model is not None:
            # This would be implemented based on your specific model architecture
            # Here's a placeholder implementation
            forecast = self._make_model_prediction(model, latest_data, days)
        else:
            # Create placeholder predictions (random walk with noise)
            last_sales = df['sales'].iloc[-1]
            forecast = [last_sales]
            for i in range(days-1):
                next_val = forecast[-1] * (1 + np.random.normal(0, 0.02))
                forecast.append(next_val)
        
        # Calculate confidence intervals based on specified level
        z_value = 1.96  # Default for 95%
        if confidence_interval == 0.90:
            z_value = 1.645
        elif confidence_interval == 0.80:
            z_value = 1.28
        
        # Estimate standard error (using historical volatility as a simple approach)
        historical_volatility = df['sales'].pct_change().std()
        error_margin = [f * historical_volatility * z_value for f in forecast]
        
        # Create forecasting chart
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(df['date'][-60:], df['sales'][-60:], label='Historical Sales')
        
        # Plot forecast
        plt.plot(future_dates, forecast, label=f'Sales Forecast ({model_type.upper()})', color='red')
        
        # Add confidence interval
        plt.fill_between(future_dates,
                         [f - e for f, e in zip(forecast, error_margin)],
                         [f + e for f, e in zip(forecast, error_margin)],
                         color='red', alpha=0.2, 
                         label=f'{int(confidence_interval*100)}% Confidence Interval')
        
        plt.title(f'Sales Forecast ({days} days ahead)')
        plt.xlabel('Date')
        plt.ylabel('Sales (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure with unique name based on parameters
        filename = f'sales_forecast_{model_type}_{days}days_{int(confidence_interval*100)}pct.png'
        output_path = os.path.join(self.reports_dir, filename)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Sales forecast report generated: {output_path}")
        
        return output_path
    
    def _make_model_prediction(self, model, latest_data, days):
        """Helper method to make prediction using the loaded model."""
        # This is a placeholder implementation
        # In a real scenario, you would prepare the features as required by your model
        
        # Start with last known value
        last_value = latest_data['sales'].iloc[-1]
        forecast = [last_value]
        
        # Make sequential predictions
        for i in range(days-1):
            # Simple placeholder: add random walk with slight upward trend
            next_val = forecast[-1] * (1 + np.random.normal(0.005, 0.02))
            forecast.append(next_val)
            
        return forecast
    
    def generate_performance_metrics(self, model_type='ffn'):
        """
        Generate performance metrics report.
        
        Parameters:
        model_type (str): Type of model to visualize metrics for ('ffn' or 'lstm')
        
        Returns:
        str: Path to the generated metrics image
        """
        metrics_path = os.path.join(os.path.dirname(self.model_path), f'{model_type}_metrics.json')
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            # Create a simple visualization of metrics
            plt.figure(figsize=(8, 6))
            metrics_names = ['mse', 'mae', 'r2', 'mape']
            metrics_values = [metrics.get(m, 0) for m in metrics_names]
            
            colors = ['lightblue', 'lightgreen', 'salmon', 'lightyellow']
            
            plt.bar(metrics_names, metrics_values, color=colors)
            plt.title(f"Model Performance Metrics\n{model_type.upper()} Model")
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Annotate bars
            for i, v in enumerate(metrics_values):
                plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
                
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(self.reports_dir, f'model_metrics_{model_type}.png')
            plt.savefig(output_path, dpi=300)
            logger.info(f"Performance metrics report generated: {output_path}")
            
            return output_path
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load model metrics for {model_type}: {e}")
            return None
    
    def generate_feature_importance(self, model_type='ffn'):
        """Generate feature importance visualization."""
        # This is a placeholder implementation
        # In a real scenario, feature importance would be extracted from the model
        
        plt.figure(figsize=(10, 6))
        
        # Sample feature importance data
        features = ['day_of_week', 'month', 'is_holiday', 'promotion_active', 'previous_day_sales']
        importance = [0.15, 0.20, 0.25, 0.30, 0.10]
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        
        # Create horizontal bar chart
        plt.barh(np.array(features)[sorted_idx], np.array(importance)[sorted_idx])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({model_type.upper()} Model)')
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.reports_dir, f'feature_importance_{model_type}.png')
        plt.savefig(output_path, dpi=300)
        logger.info(f"Feature importance report generated: {output_path}")
        
        return output_path
            
    def generate_all_reports(self):
        """Generate all reports."""
        reports = {}
        
        reports['sales_trend'] = self.generate_sales_trend_report()
        reports['category_analysis'] = self.generate_category_analysis()
        reports['weekly_pattern'] = self.generate_weekly_pattern_report()
        reports['sales_forecast'] = self.generate_sales_forecast()
        reports['performance_metrics_ffn'] = self.generate_performance_metrics('ffn')
        reports['performance_metrics_lstm'] = self.generate_performance_metrics('lstm')
        reports['feature_importance_ffn'] = self.generate_feature_importance('ffn')
        reports['feature_importance_lstm'] = self.generate_feature_importance('lstm')
        # Save report metadata
        metadata = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'reports': {k: os.path.basename(v) if v else None for k, v in reports.items()}
        }

        # Write metadata to file
        metadata_path = os.path.join(self.reports_dir, 'report_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"All reports generated successfully. Metadata saved to {metadata_path}")
        return reports