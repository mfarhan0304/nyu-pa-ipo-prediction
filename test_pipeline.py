#!/usr/bin/env python3
"""
Test script for the modified IPO analysis pipeline
Tests both regression (close price) and classification (up/down) predictions
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading and filtering"""
    try:
        logger.info("Testing data loading...")
        
        from data_loader import DataLoader
        loader = DataLoader()
        
        # Load IPO data
        ipo_data = loader.load_ipo_data()
        logger.info(f"Loaded {len(ipo_data)} IPO records")
        
        # Check filtering
        initial_count = len(ipo_data)
        
        # Check if close_price exists and is valid
        if 'close price' in ipo_data.columns:
            valid_close_price = ipo_data['close price'].notna() & (ipo_data['close price'] > 0)
            logger.info(f"Valid close prices: {valid_close_price.sum()}/{len(ipo_data)}")
            
            # Check filing filtering
            valid_filing = (ipo_data['filing'] != '0') & (ipo_data['filing'] != 0) & (ipo_data['filing'].astype(str).str.strip() != '')
            logger.info(f"Valid filings: {valid_filing.sum()}/{len(ipo_data)}")
            
            # Combined valid data
            valid_data = valid_close_price & valid_filing
            logger.info(f"Combined valid data: {valid_data.sum()}/{len(ipo_data)}")
            
            # Show sample of valid data
            sample_data = ipo_data[valid_data][['Symbol', 'Company Name', 'Price', 'close price', 'filing']].head()
            logger.info(f"Sample valid data:\n{sample_data}")
            
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering with new target variables"""
    try:
        logger.info("Testing feature engineering...")
        
        from data_loader import DataLoader
        from feature_engineer import FeatureEngineer
        
        # Load data
        loader = DataLoader()
        ipo_data = loader.load_ipo_data()
        
        # Create sample data with target variables
        sample_data = ipo_data.head(10).copy()
        
        # Add target variables
        if 'Price' in sample_data.columns and 'close price' in sample_data.columns:
            sample_data['close_price_target'] = sample_data['close price']
            sample_data['price_direction'] = np.where(sample_data['close price'] > sample_data['Price'], 1, 0)
            sample_data['first_day_return'] = ((sample_data['close price'] - sample_data['Price']) / sample_data['Price'] * 100)
            
            logger.info(f"Target variables created:")
            logger.info(f"- close_price_target range: {sample_data['close_price_target'].min():.2f} - {sample_data['close_price_target'].max():.2f}")
            logger.info(f"- price_direction distribution: {sample_data['price_direction'].value_counts().to_dict()}")
            logger.info(f"- first_day_return range: {sample_data['first_day_return'].min():.2f}% - {sample_data['first_day_return'].max():.2f}%")
        
        # Test feature engineering
        feature_engineer = FeatureEngineer()
        engineered_features = feature_engineer.engineer_features(sample_data)
        
        logger.info(f"Feature engineering completed. Shape: {engineered_features.shape}")
        logger.info(f"Features: {list(engineered_features.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization for both regression and classification"""
    try:
        logger.info("Testing model initialization...")
        
        from model_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        
        logger.info(f"Initialized {len(trainer.regression_models)} regression models")
        logger.info(f"Initialized {len(trainer.classification_models)} classification models")
        
        # Show model names
        regression_names = [info['name'] for info in trainer.regression_models.values()]
        classification_names = [info['name'] for info in trainer.classification_models.values()]
        
        logger.info(f"Regression models: {regression_names}")
        logger.info(f"Classification models: {classification_names}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model initialization test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting pipeline tests...")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_feature_engineering),
        ("Model Initialization", test_model_initialization)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info('='*50)
        
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
            logger.info(f"{test_name} test: {results[test_name]}")
        except Exception as e:
            results[test_name] = "ERROR"
            logger.error(f"{test_name} test failed with error: {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info('='*50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    all_passed = all(result == "PASSED" for result in results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The pipeline is ready to use.")
    else:
        logger.warning("\n⚠️  Some tests failed. Please check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    main()
