# Smart Retail System - Analytics & Reporting

This project implements a Business Intelligence (BI) system for retail analytics and reporting. It leverages machine learning and data analytics to provide insights into sales trends, predict demand, and generate business reports.

## Features

- **Sales Trend Analysis**: Track and visualize sales patterns over time
- **Category Analysis**: Analyze sales performance across different product categories
- **Demand Prediction**: Machine learning model to forecast future sales
- **Automated Reporting**: Generate visualizations and reports for business insights
- **Interactive Dashboard**: Web-based interface for exploring data and predictions

## Project Structure

smart_retail_system/
├── app.py                        # Flask main application file
├── config.py                     # Configuration settings
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
├── data/
│   ├── sales.csv                 # Raw sales data
│   └── processed_sales.csv       # Processed data for modeling
├── models/
│   ├── sales_model.h5            # TensorFlow model file
│   ├── scaler.pkl                # Feature scaler for data normalization
│   └── model_metadata.json       # Model performance metrics and configuration
├── reports/
│   ├── category_analysis.png         # Bar and pie charts of sales by category
│   ├── category_distribution.png     # Distribution of categories in sales
│   ├── feature_importance_ffn.png    # Feature importance for FFN model
│   ├── feature_importance_lstm.png   # Feature importance for LSTM model
│   ├── feature_importance_rf.png     # Feature importance for Random Forest model
│   ├── model_metrics_ffn.png         # FFN model evaluation metrics
│   ├── model_metrics_lstm.png        # LSTM model evaluation metrics
│   ├── model_training_history.png    # Learning curves showing loss and metrics by epoch
│   ├── performance_metrics.json      # Model performance metrics in JSON format
│   ├── performance_metrics.png       # Bar chart of model evaluation metrics
│   ├── report_metadata.json          # References to all generated reports
│   ├── sales_forecast.png            # Historical & predicted sales time series
│   ├── sales_forecast_ffn_30days_95pct.png  # FFN 30-day forecast with confidence interval
│   ├── sales_forecast_ffn_120days_95pct.png # FFN 120-day forecast with confidence interval
│   ├── sales_trend.png               # Overall sales trend analysis
│   ├── weekly_pattern.png            # Weekly sales pattern visualization
├── scripts/
│   ├── __init__.py               # Makes scripts a package
│   ├── data_processor.py         # Data processing module
│   ├── train_model.py            # Model training module
│   └── generate_report.py        # Report generation module
├── static/
│   ├── css/
│   │   └── styles.css            # Custom CSS styles
│   ├── js/
│   │   └── dashboard.js          # Dashboard JavaScript
│   └── images/                   # Image assets
├── templates/
│   ├── base.html                 # Base template with common elements
│   ├── index.html                # Landing page
│   ├── dashboard.html            # Main analytics dashboard
│   ├── predict.html              # Sales prediction page
│   └── reports.html              # Generated reports page
└── tests/
    ├── __init__.py               # Makes tests a package
    ├── test_data_processor.py    # Data processor tests
    ├── test_model.py             # Model tests
    └── test_app.py               # Flask app tests

``` 
## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/smart-retail-system.git
   cd smart-retail-system
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root directory with the following values:
   ```
   SECRET_KEY=your_secret_key
   DATABASE_URL=sqlite:///retail_data.db
   FLASK_CONFIG=development
   DEBUG=True
   ```

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Access the dashboard:**
   Open a web browser and navigate to `http://127.0.0.1:5000/`

3. **Generate sample data:**
   The system will automatically generate sample data if none exists. You can also upload your own CSV file.

4. **Train a model:**
   Navigate to the Training page and select a model type to train.

5. **Generate reports:**
   Visit the Reports page to generate and view various analytical reports.

## API Endpoints

The application provides the following API endpoints:

- `/api/data/summary` - Get summary statistics of the sales data
- `/api/data/latest` - Get the latest 30 records from the database
- `/api/predict` - Make sales predictions based on provided features

## Model Architecture

The system includes two types of prediction models:

1. **Feedforward Neural Network (FFN)**
   - Input: Sales features (day, month, year, etc.)
   - Architecture: 3 dense layers with batch normalization and dropout
   - Output: Predicted sales value

2. **LSTM (Long Short-Term Memory)**
   - Input: Sequences of sales data
   - Architecture: 2 LSTM layers with dropout
   - Output: Predicted sales value

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests
   ```bash
   pytest
   ```
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.