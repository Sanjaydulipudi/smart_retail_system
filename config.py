import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_for_development')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///retail_data.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Google Sheets API settings
    GOOGLE_API_CREDENTIALS = os.environ.get('GOOGLE_API_CREDENTIALS', 'credentials.json')
    SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID')
    
    # Model settings
    MODEL_PATH = os.path.join('models', 'sales_model.h5')
    
    # Data settings
    DATA_PATH = os.path.join('data', 'sales.csv')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed_sales.csv')
    
    # Reporting settings
    REPORTS_DIR = 'static/reports/'
    
    # Debug settings
    DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test_retail_data.db'


# Configuration dictionary
config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Get configuration based on environment
def get_config():
    config_name = os.environ.get('FLASK_CONFIG', 'default')
    return config_dict[config_name]