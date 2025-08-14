"""
Configuration file for IPO Analysis ML Pipeline
===============================================
Centralized configuration for all pipeline parameters and settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data file paths
IPO_DATA_PATH = DATA_DIR / "ipo_details_enriched.csv"
VIX_DATA_PATH = DATA_DIR / "vix_2000_to_2025.csv"
FEDFUNDS_DATA_PATH = DATA_DIR / "FEDFUNDS.csv"
SEC_FILINGS_DIR = DATA_DIR / "sec-edgar-filings"

# Output file paths
SEC_FEATURES_PATH = RESULTS_DIR / "sec_filing_features.csv"
ENHANCED_DATASET_PATH = RESULTS_DIR / "enhanced_ipo_dataset.csv"
MODEL_RESULTS_PATH = RESULTS_DIR / "regression_results.csv"
CLASSIFICATION_RESULTS_PATH = RESULTS_DIR / "classification_results.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
REGRESSION_REPORT_PATH = RESULTS_DIR / "regression_report.txt"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"

# ML Pipeline Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5

# Model Configuration
MODELS_CONFIG = {
    'linear_regression': {
        'name': 'Linear Regression',
        'class': 'sklearn.linear_model.LinearRegression',
        'params': {}
    },
    'ridge_regression': {
        'name': 'Ridge Regression',
        'class': 'sklearn.linear_model.Ridge',
        'params': {'alpha': 1.0}
    },
    'random_forest': {
        'name': 'Random Forest',
        'class': 'sklearn.ensemble.RandomForestRegressor',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'class': 'sklearn.ensemble.GradientBoostingRegressor',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_STATE
        }
    },
    'xgboost': {
        'name': 'XGBoost',
        'class': 'xgboost.XGBRegressor',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_STATE
        }
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'traditional_features': [
        'Price', 'Shares', 'Employees', 'Total Offering Expense',
        'Lockup Period (days)', 'Quiet Period (days)'
    ],
    'nlp_features': [
        'text_word_count', 'text_sentence_count', 'text_complexity', 'text_diversity',
        'avg_positive_sentiment', 'risk_sentiment_score', 'risk_intensity',
        'tech_innovation_score', 'market_awareness', 'regulatory_compliance',
        'growth_potential', 'document_quality', 'filing_type_encoded'
    ],
    'financial_features_prefixes': ['financial_'],
    'section_features_prefixes': ['section_'],
    'embedding_features_prefixes': ['embedding_']
}

# NLP Configuration
NLP_CONFIG = {
    'max_features': 500,
    'max_text_length': 500000,  # 500KB limit for performance
    'sentiment_text_limit': 2000,
    'batch_size': 8,
    'financial_keywords': {
        'risk': ['risk', 'uncertainty', 'volatility', 'exposure', 'liability', 'danger', 'threat'],
        'growth': ['growth', 'expansion', 'increase', 'development', 'acquisition', 'scaling'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'profit', 'turnover', 'gross'],
        'debt': ['debt', 'liability', 'obligation', 'loan', 'credit', 'borrowing', 'leverage'],
        'market': ['market', 'competition', 'industry', 'sector', 'demand', 'customer', 'client'],
        'technology': ['technology', 'innovation', 'patent', 'intellectual property', 'software', 'digital'],
        'regulation': ['regulation', 'compliance', 'legal', 'regulatory', 'approval', 'license', 'permit']
    },
    'section_patterns': {
        'business_description': r'(?i)(item\s*1\.?\s*|business\s*description|description\s*of\s*business).*?(?=item\s*2|item\s*1a|$)',
        'risk_factors': r'(?i)(item\s*1a\.?\s*|risk\s*factors).*?(?=item\s*2|item\s*1b|$)',
        'management_discussion': r'(?i)(item\s*7\.?\s*|management\s*discussion|md&a).*?(?=item\s*8|item\s*7a|$)',
        'financial_statements': r'(?i)(item\s*8\.?\s*|financial\s*statements).*?(?=item\s*9|item\s*8a|$)',
        'executive_compensation': r'(?i)(item\s*11\.?\s*|executive\s*compensation).*?(?=item\s*12|item\s*11a|$)',
        'legal_proceedings': r'(?i)(item\s*3\.?\s*|legal\s*proceedings).*?(?=item\s*4|item\s*3a|$)'
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'pipeline.log'
}

# Visualization Configuration
VIZ_CONFIG = {
    'style': 'default',
    'figsize': (18, 12),
    'dpi': 300,
    'save_format': 'png'
}
