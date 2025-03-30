from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

# Import local modules
from config import get_config
from scripts.data_processor import DataProcessor
from scripts.train_model import SalesPredictionModel
from scripts.generate_report import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler("app.log")
                    ])
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(get_config())
app.secret_key = app.config.get('SECRET_KEY', 'dev_key')

# Ensure directories exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'reports'), exist_ok=True)

# Update reports directory to be in static folder
app.config['REPORTS_DIR'] = os.path.join(os.path.dirname(__file__), 'static', 'reports')

# Context processor for template variables
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Global variables
DATA_PROCESSOR = DataProcessor()
MODEL = SalesPredictionModel()
REPORT_GENERATOR = ReportGenerator()

@app.route('/')
def home():
    """Render home page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render main analytics dashboard."""
    # Get report metadata
    reports_dir = app.config.get('REPORTS_DIR')
    metadata_path = os.path.join(reports_dir, 'report_metadata.json')
    
    reports = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            try:
                metadata = json.load(f)
                reports = metadata.get('reports', {})
                # Add full paths to reports
                reports = {k: os.path.join('reports', v) if v else None 
                          for k, v in reports.items()}
            except json.JSONDecodeError:
                flash("Error loading report metadata", "error")
    
    # Generate reports if they don't exist
    if not reports:
        try:
            generated_reports = REPORT_GENERATOR.generate_all_reports()
            # Add full paths to reports
            reports = {k: os.path.join('reports', os.path.basename(v)) if v else None 
                      for k, v in generated_reports.items()}
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            flash(f"Error generating reports: {str(e)}", "error")
    
    # Get summary statistics
    try:
        df = DATA_PROCESSOR.load_data()
        summary = {
            'total_sales': f"${df['sales'].sum():,.2f}",
            'avg_daily_sales': f"${df['sales'].mean():,.2f}",
            'sales_growth': f"{(df['sales'].iloc[-1] / df['sales'].iloc[0] - 1) * 100:.1f}%",
            'data_period': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'record_count': f"{len(df):,}"
        }
    except Exception as e:
        logger.error(f"Error loading data summary: {e}")
        summary = {
            'total_sales': 'N/A',
            'avg_daily_sales': 'N/A',
            'sales_growth': 'N/A',
            'data_period': 'N/A',
            'record_count': 'N/A'
        }
    
    return render_template('dashboard.html', reports=reports, summary=summary)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Render prediction page and handle prediction requests."""
    if request.method == 'POST':
        try:
            # Get form data
            prediction_days = int(request.form.get('prediction_days', 30))
            model_type = request.form.get('model_type', 'linear')
            confidence_interval = float(request.form.get('confidence_interval', 0.95))
            
            # Generate forecast
            forecast_path = REPORT_GENERATOR.generate_sales_forecast(
                days=prediction_days,
                model_type=model_type,
                confidence_interval=confidence_interval
            )
            
            # Get model metrics if available
            metrics = None
            feature_importance = None
            try:
                metrics_path = os.path.join(app.config.get('MODELS_DIR'), f"{model_type}_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                
                # Check for feature importance
                # fi_path = os.path.join(app.config.get('REPORTS_DIR'), f"feature_importance_{model_type}.png")
                fi_path = os.path.join('/static/reports/', f"feature_importance_{model_type}.png")
                if os.path.exists(fi_path):
                    feature_importance = os.path.join('reports', os.path.basename(fi_path))
            except Exception as e:
                logger.warning(f"Could not load model metrics: {e}")
            
            return render_template('predict.html', 
                                  forecast_image=os.path.basename(forecast_path),
                                  prediction_days=prediction_days,
                                  model_type=model_type,
                                  confidence_interval=confidence_interval,
                                  metrics=metrics,
                                  feature_importance=feature_importance)
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            flash(f"Error generating prediction: {str(e)}", "error")
            return render_template('predict.html')
    else:
        return render_template('predict.html')
    
@app.route('/reports')
def reports():
    """Render reports page."""
    # Get available reports
    reports_dir = app.config.get('REPORTS_DIR')
    available_reports = []
    
    if os.path.exists(reports_dir):
        for filename in os.listdir(reports_dir):
            if filename.endswith('.png'):
                report_path = os.path.join('reports', filename)
                report_name = ' '.join(filename.replace('.png', '').split('_')).title()
                available_reports.append({
                    'name': report_name,
                    'path': report_path,
                    'date': datetime.fromtimestamp(os.path.getmtime(os.path.join(reports_dir, filename)))
                             .strftime('%Y-%m-%d %H:%M')
                })
    
    return render_template('reports.html', reports=available_reports)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Render model training page and handle training requests."""
    if request.method == 'POST':
        try:
            # Get form data
            model_type = request.form.get('model_type', 'linear')
            
            # Validate model type
            if model_type not in ['linear', 'rf', 'time_series']:
                flash('Invalid model type', 'error')
                return render_template('train.html', success=False)
            
            # Train model
            metrics = MODEL.train_model(model_type=model_type)
            
            # Generate performance report
            try:
                perf_report = REPORT_GENERATOR.generate_performance_metrics()
            except Exception as e:
                logger.warning(f"Could not generate performance report: {e}")
                perf_report = None
            
            return render_template('train.html', 
                                  success=True,
                                  metrics=metrics,
                                  perf_report=os.path.basename(perf_report) if perf_report else None)
        except Exception as e:
            logger.error(f"Error training model: {e}")
            flash(f"Error training model: {str(e)}", "error")
            return render_template('train.html', success=False)
    else:
        return render_template('train.html', success=False)

@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    """Render data upload page and handle file uploads."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config.get('DATA_PATH'), filename)
                
                # Save file
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
                
                # Process data
                DATA_PROCESSOR.process_data()
                
                flash('File uploaded and processed successfully', 'success')
                return redirect(url_for('dashboard'))
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                flash(f"Error processing file: {str(e)}", "error")
        else:
            flash('Only CSV files are allowed', 'error')
            
    return render_template('upload.html')

@app.route('/api/data/summary')
def api_data_summary():
    """API endpoint for data summary."""
    try:
        df = DATA_PROCESSOR.load_data()
        
        # Calculate summary statistics
        summary = {
            'total_sales': float(df['sales'].sum()),
            'avg_daily_sales': float(df['sales'].mean()),
            'max_sales': float(df['sales'].max()),
            'min_sales': float(df['sales'].min()),
            'sales_std': float(df['sales'].std()),
            'record_count': len(df)
        }
        
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data/latest')
def api_data_latest():
    """API endpoint for latest data."""
    try:
        df = DATA_PROCESSOR.load_data()
        
        # Get the latest 30 records
        latest = df.tail(30).to_dict(orient='records')
        
        return jsonify({'success': True, 'data': latest})
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions."""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        features = pd.DataFrame([data])
        
        # Make prediction
        prediction = MODEL.predict(features)
        print('predictions:',prediction)
        return jsonify({
            'success': True, 
            'prediction': float(prediction[0])
        })
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def ensure_data_exists():
    """Ensure data directory exists and sample data is generated if needed."""
    # Ensure data directory exists
    data_path = app.config.get('DATA_PATH')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Generate sample data if it doesn't exist
    if not os.path.exists(data_path):
        logger.info("Generating sample data...")
        DATA_PROCESSOR.generate_sample_data()
        DATA_PROCESSOR.process_data()

if __name__ == '__main__':
    # Ensure data and directories exist
    ensure_data_exists()
    
    # Install flask-cors if not already installed
    try:
        from flask_cors import CORS
        CORS(app)  # Enable CORS for all routes
        logger.info("CORS enabled for all routes")
    except ImportError:
        logger.warning("flask-cors not installed. Cross-origin requests may be blocked.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config.get('DEBUG', False))