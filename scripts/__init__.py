"""
Smart Retail System - Scripts Package

This package contains modules for data processing, model training, and report generation.
"""

# Import modules to make them available when importing the package
from scripts.data_processor import DataProcessor
from scripts.train_model import SalesPredictionModel
from scripts.generate_report import ReportGenerator

__all__ = ['DataProcessor', 'SalesPredictionModel', 'ReportGenerator']