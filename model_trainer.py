"""
Model Trainer Module for IPO Analysis ML Pipeline
================================================
Handles model training, evaluation, and comparison for the IPO analysis pipeline
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb

from config import MODELS_CONFIG, RANDOM_STATE, TEST_SIZE, CROSS_VALIDATION_FOLDS

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model training and evaluation class for IPO analysis pipeline
    """
    
    def __init__(self):
        """Initialize the model trainer"""
        self.regression_models = {}
        self.classification_models = {}
        self.trained_regression_models = {}
        self.trained_classification_models = {}
        self.regression_results = {}
        self.classification_results = {}
        self.best_regression_model = None
        self.best_classification_model = None
        self.best_regression_model_name = None
        self.best_classification_model_name = None
        
        # Initialize models
        self._initialize_models()
        
        logger.info("ModelTrainer initialized")
    
    def _initialize_models(self):
        """Initialize all models from configuration"""
        try:
            logger.info("Initializing models...")
            
            # Initialize regression models
            regression_configs = {
                'linear_regression': {'name': 'Linear Regression', 'class': 'sklearn.linear_model.LinearRegression', 'params': {}},
                'ridge_regression': {'name': 'Ridge Regression', 'class': 'sklearn.linear_model.Ridge', 'params': {'alpha': 1.0}},
                'random_forest_regression': {'name': 'Random Forest Regression', 'class': 'sklearn.ensemble.RandomForestRegressor', 'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE}},
                'gradient_boosting_regression': {'name': 'Gradient Boosting Regression', 'class': 'sklearn.ensemble.GradientBoostingRegressor', 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': RANDOM_STATE}},
                'xgboost_regression': {'name': 'XGBoost Regression', 'class': 'xgboost.XGBRegressor', 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': RANDOM_STATE}}
            }
            
            # Initialize classification models
            classification_configs = {
                'logistic_regression': {'name': 'Logistic Regression', 'class': 'sklearn.linear_model.LogisticRegression', 'params': {'random_state': RANDOM_STATE}},
                'random_forest_classification': {'name': 'Random Forest Classification', 'class': 'sklearn.ensemble.RandomForestClassifier', 'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE}},
                'gradient_boosting_classification': {'name': 'Gradient Boosting Classification', 'class': 'sklearn.ensemble.GradientBoostingClassifier', 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': RANDOM_STATE}},
                'xgboost_classification': {'name': 'XGBoost Classification', 'class': 'xgboost.XGBClassifier', 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': RANDOM_STATE}}
            }
            
            # Initialize regression models
            for model_key, model_config in regression_configs.items():
                try:
                    module_path, class_name = model_config['class'].rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    model_class = getattr(module, class_name)
                    model = model_class(**model_config['params'])
                    
                    self.regression_models[model_key] = {
                        'name': model_config['name'],
                        'instance': model,
                        'params': model_config['params']
                    }
                    logger.info(f"Initialized {model_config['name']}")
                except Exception as e:
                    logger.warning(f"Could not initialize {model_key}: {e}")
                    continue
            
            # Initialize classification models
            for model_key, model_config in classification_configs.items():
                try:
                    module_path, class_name = model_config['class'].rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    model_class = getattr(module, class_name)
                    model = model_class(**model_config['params'])
                    
                    self.classification_models[model_key] = {
                        'name': model_config['name'],
                        'instance': model,
                        'params': model_config['params']
                    }
                    logger.info(f"Initialized {model_config['name']}")
                except Exception as e:
                    logger.warning(f"Could not initialize {model_key}: {e}")
                    continue
            
            logger.info(f"Successfully initialized {len(self.regression_models)} regression models and {len(self.classification_models)} classification models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            logger.info("Preparing data for training...")
            
            # Remove rows with invalid target values
            valid_mask = np.isfinite(y) & (y != 0)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            logger.info(f"Data shape after cleaning: {X_clean.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, 
                test_size=TEST_SIZE, 
                random_state=RANDOM_STATE
            )
            
            logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_regression_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train all regression models
        
        Args:
            X: Feature matrix
            y: Regression target
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Training regression models...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            
            logger.info(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
            
            results = {}
            
            for model_key, model_info in self.regression_models.items():
                try:
                    logger.info(f"Training {model_info['name']}...")
                    
                    # Train model
                    model = model_info['instance']
                    model.fit(X_train, y_train)
                    
                    # Store trained model
                    self.trained_regression_models[model_key] = model
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_metrics = self._calculate_metrics(y_train, y_train_pred, 'train')
                    test_metrics = self._calculate_metrics(y_test, y_test_pred, 'test')
                    metrics = {**train_metrics, **test_metrics}
                    
                    # Store results
                    results[model_key] = {
                        'name': model_info['name'],
                        'metrics': metrics,
                        'predictions': {
                            'train': y_train_pred,
                            'test': y_test_pred
                        }
                    }
                    
                    logger.info(f"{model_info['name']}: Train R² = {metrics['train_r2']:.4f}, Test R² = {metrics['test_r2']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_key}: {e}")
                    continue
            
            # Store results
            self.regression_results = results
            
            # Find best model
            self._find_best_model()
            
            logger.info("Regression model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in regression model training: {e}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names
            
        Returns:
            Dictionary with metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics[f'{prefix}_mse'] = mean_squared_error(y_true, y_pred)
            metrics[f'{prefix}_rmse'] = np.sqrt(metrics[f'{prefix}_mse'])
            metrics[f'{prefix}_mae'] = mean_absolute_error(y_true, y_pred)
            metrics[f'{prefix}_r2'] = r2_score(y_true, y_pred)
            
            # Additional metrics
            metrics[f'{prefix}_mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _find_best_model(self):
        """Find the best performing model"""
        try:
            if not self.regression_results:
                return
            
            # Find best model based on test R² score
            best_score = -np.inf
            best_model_key = None
            
            for model_key, result in self.regression_results.items():
                if 'test_r2' in result['metrics']:
                    score = result['metrics']['test_r2']
                    if score > best_score:
                        best_score = score
                        best_model_key = model_key
            
            if best_model_key:
                self.best_regression_model = self.trained_regression_models[best_model_key]
                self.best_regression_model_name = self.regression_results[best_model_key]['name']
                logger.info(f"Best regression model: {self.best_regression_model_name} (R² = {best_score:.4f})")
            else:
                # Fallback to training R² if no test metrics
                best_score = -np.inf
                for model_key, result in self.regression_results.items():
                    score = result['metrics']['train_r2']
                    if score > best_score:
                        best_score = score
                        best_model_key = model_key
                
                if best_model_key:
                    self.best_regression_model = self.trained_regression_models[best_model_key]
                    self.best_regression_model_name = self.regression_results[best_model_key]['name']
                    logger.info(f"Best regression model (train): {self.best_regression_model_name} (R² = {best_score:.4f})")
                    
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform cross-validation for all models
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            logger.info("Starting cross-validation...")
            
            cv_results = {}
            
            for model_key, model_info in self.regression_models.items():
                try:
                    logger.info(f"Cross-validating {model_info['name']}...")
                    
                    model = model_info['instance']
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(
                        model, X, y, 
                        cv=CROSS_VALIDATION_FOLDS, 
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    # Calculate statistics
                    cv_results[model_key] = {
                        'name': model_info['name'],
                        'cv_scores': cv_scores,
                        'mean_score': cv_scores.mean(),
                        'std_score': cv_scores.std(),
                        'min_score': cv_scores.min(),
                        'max_score': cv_scores.max()
                    }
                    
                    logger.info(f"  {model_info['name']}: CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in cross-validation for {model_key}: {e}")
                    continue
            
            logger.info("Cross-validation completed successfully")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                             model_key: str, param_grid: Dict) -> Dict:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            X_train: Training features
            y_train: Training target
            model_key: Key of the model to tune
            param_grid: Parameter grid for tuning
            
        Returns:
            Dictionary with tuning results
        """
        try:
            if model_key not in self.regression_models:
                raise ValueError(f"Regression model {model_key} not found")
            
            logger.info(f"Performing hyperparameter tuning for {self.regression_models[model_key]['name']}...")
            
            model = self.regression_models[model_key]['instance']
            
            # Perform grid search
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=CROSS_VALIDATION_FOLDS,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best parameters and score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # Update model with best parameters
            best_model = grid_search.best_estimator_
            self.trained_regression_models[model_key] = best_model
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best CV score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'best_model': best_model,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise
    
    def get_feature_importance(self, model_key: str = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            model_key: Specific model key (if None, use best model)
            
        Returns:
            DataFrame with feature importance
        """
        try:
            if model_key is None:
                if self.best_regression_model is None:
                    logger.warning("No best regression model available")
                    return pd.DataFrame()
                model = self.best_regression_model
                model_name = self.best_regression_model_name
            else:
                if model_key not in self.trained_regression_models:
                    logger.warning(f"Regression model {model_key} not found")
                    return pd.DataFrame()
                model = self.trained_regression_models[model_key]
                model_name = self.regression_models[model_key]['name']
            
            # Check if model supports feature importance
            if not hasattr(model, 'feature_importances_'):
                logger.warning(f"Model {model_name} does not support feature importance")
                return pd.DataFrame()
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': range(len(importance)),
                'importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            logger.info(f"Feature importance calculated for {model_name}")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def make_predictions(self, X: np.ndarray, model_key: str = None) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            X: Feature matrix
            model_key: Specific model key (if None, use best model)
            
        Returns:
            Array of predictions
        """
        try:
            if model_key is None:
                if self.best_regression_model is None:
                    raise ValueError("No best regression model available")
                model = self.best_regression_model
            else:
                if model_key not in self.trained_regression_models:
                    raise ValueError(f"Regression model {model_key} not found")
                model = self.trained_regression_models[model_key]
            
            # Make predictions
            predictions = model.predict(X)
            
            logger.info(f"Predictions made using {model_key or 'best regression model'}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of all trained models (both regression and classification)
        
        Returns:
            Dictionary with comprehensive model summary
        """
        summary = {
            # Overall counts
            'total_models': len(self.regression_models) + len(self.classification_models),
            'trained_models': len(self.trained_regression_models) + len(self.trained_classification_models),
            
            # Regression models
            'total_regression_models': len(self.regression_models),
            'trained_regression_models': len(self.trained_regression_models),
            'best_regression_model': self.best_regression_model_name,
            'best_regression_score': None,
            
            # Classification models
            'total_classification_models': len(self.classification_models),
            'trained_classification_models': len(self.trained_classification_models),
            'best_classification_model': self.best_classification_model_name,
            'best_classification_score': None,
            
            # Best overall model
            'best_model': None,
            'best_model_type': None,
            'best_model_score': None,
            
            # Detailed results
            'regression_results': {},
            'classification_results': {}
        }
        
        # Add regression results
        for model_key, result in self.regression_results.items():
            summary['regression_results'][model_key] = {
                'name': result['name'],
                'train_r2': result['metrics'].get('train_r2', None),
                'test_r2': result['metrics'].get('test_r2', None),
                'train_rmse': result['metrics'].get('train_rmse', None),
                'test_rmse': result['metrics'].get('test_rmse', None)
            }
        
        # Add classification results
        for model_key, result in self.classification_results.items():
            summary['classification_results'][model_key] = {
                'name': result['name'],
                'train_accuracy': result['metrics'].get('train_accuracy', None),
                'test_accuracy': result['metrics'].get('test_accuracy', None),
                'train_f1': result['metrics'].get('train_f1', None),
                'test_f1': result['metrics'].get('test_f1', None)
            }
        
        # Determine best models and scores
        if self.regression_results:
            best_reg_score = max([result['metrics'].get('test_r2', 0) for result in self.regression_results.values()])
            summary['best_regression_score'] = best_reg_score
            
        if self.classification_results:
            best_clf_score = max([result['metrics'].get('test_accuracy', 0) for result in self.classification_results.values()])
            summary['best_classification_score'] = best_clf_score
        
        # Determine overall best model
        if summary['best_regression_score'] is not None and summary['best_classification_score'] is not None:
            if summary['best_regression_score'] > summary['best_classification_score']:
                summary['best_model'] = summary['best_regression_model']
                summary['best_model_type'] = 'regression'
                summary['best_model_score'] = summary['best_regression_score']
            else:
                summary['best_model'] = summary['best_classification_model']
                summary['best_model_type'] = 'classification'
                summary['best_model_score'] = summary['best_classification_score']
        elif summary['best_regression_score'] is not None:
            summary['best_model'] = summary['best_regression_model']
            summary['best_model_type'] = 'regression'
            summary['best_model_score'] = summary['best_regression_score']
        elif summary['best_classification_score'] is not None:
            summary['best_model'] = summary['best_classification_model']
            summary['best_model_type'] = 'classification'
            summary['best_model_score'] = summary['best_classification_score']
        
        return summary
    
    def save_results(self, filepath: str = None):
        """
        Save all results to CSV (for backward compatibility)
        
        Args:
            filepath: Path to save file (optional)
        """
        # Save regression results
        self.save_regression_results()
        
        # Save classification results
        self.save_classification_results()
        
        logger.info("All results saved successfully")
    
    def save_regression_results(self, filepath: str = None):
        """
        Save regression results to CSV
        
        Args:
            filepath: Path to save file (optional)
        """
        if not self.regression_results:
            logger.warning("No regression results to save")
            return
        
        if filepath is None:
            from config import MODEL_RESULTS_PATH
            filepath = MODEL_RESULTS_PATH
        
        try:
            # Prepare results for saving
            results_data = []
            for model_key, result in self.regression_results.items():
                row = {
                    'model_key': model_key,
                    'name': result['name'],
                    'train_r2': result['metrics'].get('train_r2', None),
                    'test_r2': result['metrics'].get('test_r2', None),
                    'train_mse': result['metrics'].get('train_mse', None),
                    'test_mse': result['metrics'].get('test_mse', None),
                    'train_mae': result['metrics'].get('train_mae', None),
                    'test_mae': result['metrics'].get('test_mae', None)
                }
                results_data.append(row)
            
            # Create DataFrame and save
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(filepath, index=False)
            
            logger.info(f"Regression results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving regression results: {e}")
            raise

    def train_classification_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train all classification models
        
        Args:
            X: Feature matrix
            y: Classification target (0 or 1)
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Training classification models...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            
            logger.info(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
            logger.info(f"Target distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
            
            results = {}
            
            for model_key, model_info in self.classification_models.items():
                try:
                    logger.info(f"Training {model_info['name']}...")
                    
                    # Train model
                    model = model_info['instance'].fit(X_train, y_train)
                    
                    # Store trained model
                    self.trained_classification_models[model_key] = model
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    
                    # Store results
                    results[model_key] = {
                        'name': model_info['name'],
                        'metrics': {
                            'train_accuracy': train_accuracy,
                            'test_accuracy': test_accuracy
                        },
                        'predictions': {
                            'train': y_train_pred,
                            'test': y_test_pred
                        }
                    }
                    
                    logger.info(f"{model_info['name']}: Train Accuracy={train_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_key}: {e}")
                    continue
            
            # Store results
            self.classification_results = results
            
            # Find best model
            self._find_best_classification_model()
            
            logger.info("Classification model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in classification model training: {e}")
            raise
    
    def _find_best_classification_model(self):
        """Find the best performing classification model"""
        try:
            if not self.classification_results:
                return
            
            best_score = -np.inf
            best_model_key = None
            
            for model_key, result in self.classification_results.items():
                if 'test_accuracy' in result['metrics']:
                    score = result['metrics']['test_accuracy']
                    if score > best_score:
                        best_score = score
                        best_model_key = model_key
            
            if best_model_key:
                self.best_classification_model = self.trained_classification_models[best_model_key]
                self.best_classification_model_name = self.classification_results[best_model_key]['name']
                logger.info(f"Best classification model: {self.best_classification_model_name} (Accuracy = {best_score:.4f})")
            else:
                # Fallback to training accuracy if no test metrics
                best_score = -np.inf
                for model_key, result in self.classification_results.items():
                    score = result['metrics']['train_accuracy']
                    if score > best_score:
                        best_score = score
                        best_model_key = model_key
                
                if best_model_key:
                    self.best_classification_model = self.trained_classification_models[best_model_key]
                    self.best_classification_model_name = self.classification_results[best_model_key]['name']
                    logger.info(f"Best classification model (train): {self.best_classification_model_name} (Accuracy = {best_score:.4f})")
                    
        except Exception as e:
            logger.error(f"Error finding best classification model: {e}")
    
    def cross_validate_classification_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform cross-validation for all classification models
        
        Args:
            X: Feature matrix
            y: Classification target
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            logger.info("Performing cross-validation for classification models...")
            
            cv_results = {}
            
            for model_key, model_info in self.classification_models.items():
                try:
                    logger.info(f"Cross-validating {model_info['name']}...")
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(
                        model_info['instance'], X, y, 
                        cv=CROSS_VALIDATION_FOLDS, scoring='accuracy'
                    )
                    
                    cv_results[model_key] = {
                        'name': model_info['name'],
                        'cv_scores': cv_scores,
                        'mean_accuracy': cv_scores.mean(),
                        'std_accuracy': cv_scores.std()
                    }
                    
                    logger.info(f"{model_info['name']}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                    
                except Exception as e:
                    logger.error(f"Error in cross-validation for {model_key}: {e}")
                    continue
            
            logger.info("Classification cross-validation completed")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in classification cross-validation: {e}")
            raise
    
    def predict_classification(self, X: np.ndarray, model_key: str = None) -> np.ndarray:
        """
        Make classification predictions
        
        Args:
            X: Feature matrix
            model_key: Specific model to use (optional)
            
        Returns:
            Classification predictions (0 or 1)
        """
        try:
            if model_key is None:
                if self.best_classification_model is None:
                    raise ValueError("No best classification model available")
                model = self.best_classification_model
            else:
                if model_key not in self.trained_classification_models:
                    raise ValueError(f"Classification model {model_key} not found")
                model = self.trained_classification_models[model_key]
            
            # Make predictions
            predictions = model.predict(X)
            
            logger.info(f"Classification predictions made using {model_key or 'best classification model'}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in classification prediction: {e}")
            raise
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> str:
        """
        Generate classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Classification report string
        """
        try:
            report = f"Classification Report for {model_name}\n"
            report += "=" * 50 + "\n"
            report += classification_report(y_true, y_pred, target_names=['Down', 'Up'])
            report += "\nConfusion Matrix:\n"
            report += str(confusion_matrix(y_true, y_pred))
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return f"Error generating report: {e}"
    
    def save_classification_results(self, filepath: str = None):
        """
        Save classification results to CSV
        
        Args:
            filepath: Path to save file (optional)
        """
        if not self.classification_results:
            logger.warning("No classification results to save")
            return
        
        if filepath is None:
            from config import RESULTS_DIR
            filepath = RESULTS_DIR / "classification_results.csv"
        
        try:
            # Prepare results for saving
            results_data = []
            for model_key, result in self.classification_results.items():
                row = {
                    'model_key': model_key,
                    'name': result['name'],
                    'train_accuracy': result['metrics']['train_accuracy'],
                    'test_accuracy': result['metrics']['test_accuracy']
                }
                results_data.append(row)
            
            # Create DataFrame and save
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(filepath, index=False)
            
            logger.info(f"Classification results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving classification results: {e}")
            raise

    def cross_validate_regression_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform cross-validation for all regression models
        
        Args:
            X: Feature matrix
            y: Regression target
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            logger.info("Performing cross-validation for regression models...")
            
            cv_results = {}
            
            for model_key, model_info in self.regression_models.items():
                try:
                    logger.info(f"Cross-validating {model_info['name']}...")
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(
                        model_info['instance'], X, y, 
                        cv=CROSS_VALIDATION_FOLDS, scoring='r2'
                    )
                    
                    cv_results[model_key] = {
                        'name': model_info['name'],
                        'cv_scores': cv_scores,
                        'mean_r2': cv_scores.mean(),
                        'std_r2': cv_scores.std()
                    }
                    
                    logger.info(f"{model_info['name']}: CV R² = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                    
                except Exception as e:
                    logger.error(f"Error in cross-validation for {model_key}: {e}")
                    continue
            
            logger.info("Regression cross-validation completed")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in regression cross-validation: {e}")
            raise
