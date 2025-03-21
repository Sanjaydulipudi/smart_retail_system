"""
Smart Retail System - Tests Package

This package contains test modules for unit testing the application components.
"""

# Import test modules to make them discoverable by pytest
from tests.test_data_processor import TestDataProcessor
from tests.test_model import TestSalesPredictionModel
from tests.test_app import TestFlaskApp

__all__ = ['TestDataProcessor', 'TestSalesPredictionModel', 'TestFlaskApp']