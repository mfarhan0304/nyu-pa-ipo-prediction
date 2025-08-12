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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
import xgboost as xgb

from config import MODELS_CONFIG, RANDOM_STATE, TEST_SIZE, CROSS_VALIDATION_FOLDS

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model training and evaluation class for IPO analysis pipeline
    """
    
    def __init__(self):
        """Initialize the model trainer"""
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialize models
        self._initialize_models()
        
        logger.info("ModelTrainer initialized")
    
    def _initialize_models(self):
        """Initialize all models from configuration"""
        try:
            logger.info("Initializing models...")
            
            for model_key, model_config in MODELS_CONFIG.items():
                try:
                    # Import model class
                    module_path, class_name = model_config['class'].rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    model_class = getattr(module, class_name)
                    
                    # Create model instance
                    model = model_class(**model_config['params'])
                    
                    # Store model
                    self.models[model_key] = {
                        'name': model_config['name'],
                        'instance': model,
                        'params': model_config['params']
                    }
                    
                    logger.info(f"Initialized {model_config['name']}")
                    
                except Exception as e:
                    logger.warning(f"Could not initialize {model_key}: {e}")
                    continue
            
            logger.info(f"Successfully initialized {len(self.models)} models")
            
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
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray = None, y_test: np.ndarray = None) -> Dict:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Starting model training...")
            
            results = {}
            
            for model_key, model_info in self.models.items():
                try:
                    logger.info(f"Training {model_info['name']}...")
                    
                    # Train model
                    model = model_info['instance']
                    model.fit(X_train, y_train)
                    
                    # Store trained model
                    self.trained_models[model_key] = model
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test) if X_test is not None else None
                    
                    # Calculate metrics
                    train_metrics = self._calculate_metrics(y_train, y_train_pred, 'train')
                    
                    if X_test is not None and y_test is not None:
                        test_metrics = self._calculate_metrics(y_test, y_test_pred, 'test')
                        metrics = {**train_metrics, **test_metrics}
                    else:
                        metrics = train_metrics
                    
                    # Store results
                    results[model_key] = {
                        'name': model_info['name'],
                        'metrics': metrics,
                        'predictions': {
                            'train': y_train_pred,
                            'test': y_test_pred if y_test is not None else None
                        }
                    }
                    
                    logger.info(f"  {model_info['name']}: Train R² = {metrics['train_r2']:.4f}")
                    if 'test_r2' in metrics:
                        logger.info(f"  {model_info['name']}: Test R² = {metrics['test_r2']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_key}: {e}")
                    continue
            
            # Store results
            self.model_results = results
            
            # Find best model
            self._find_best_model()
            
            logger.info("Model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
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
            if not self.model_results:
                return
            
            # Find best model based on test R² score
            best_score = -np.inf
            best_model_key = None
            
            for model_key, result in self.model_results.items():
                if 'test_r2' in result['metrics']:
                    score = result['metrics']['test_r2']
                    if score > best_score:
                        best_score = score
                        best_model_key = model_key
            
            if best_model_key:
                self.best_model = self.trained_models[best_model_key]
                self.best_model_name = self.model_results[best_model_key]['name']
                logger.info(f"Best model: {self.best_model_name} (R² = {best_score:.4f})")
            else:
                # Fallback to training R² if no test metrics
                best_score = -np.inf
                for model_key, result in self.model_results.items():
                    score = result['metrics']['train_r2']
                    if score > best_score:
                        best_score = score
                        best_model_key = model_key
                
                if best_model_key:
                    self.best_model = self.trained_models[best_model_key]
                    self.best_model_name = self.model_results[best_model_key]['name']
                    logger.info(f"Best model (train): {self.best_model_name} (R² = {best_score:.4f})")
                    
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
            
            for model_key, model_info in self.models.items():
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
            if model_key not in self.models:
                raise ValueError(f"Model {model_key} not found")
            
            logger.info(f"Performing hyperparameter tuning for {self.models[model_key]['name']}...")
            
            model = self.models[model_key]['instance']
            
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
            self.trained_models[model_key] = best_model
            
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
                if self.best_model is None:
                    logger.warning("No best model available")
                    return pd.DataFrame()
                model = self.best_model
                model_name = self.best_model_name
            else:
                if model_key not in self.trained_models:
                    logger.warning(f"Model {model_key} not found")
                    return pd.DataFrame()
                model = self.trained_models[model_key]
                model_name = self.models[model_key]['name']
            
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
                if self.best_model is None:
                    raise ValueError("No best model available")
                model = self.best_model
            else:
                if model_key not in self.trained_models:
                    raise ValueError(f"Model {model_key} not found")
                model = self.trained_models[model_key]
            
            # Make predictions
            predictions = model.predict(X)
            
            logger.info(f"Predictions made using {model_key or 'best model'}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of all trained models
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'total_models': len(self.models),
            'trained_models': len(self.trained_models),
            'best_model': self.best_model_name,
            'model_results': {}
        }
        
        # Add results for each model
        for model_key, result in self.model_results.items():
            summary['model_results'][model_key] = {
                'name': result['name'],
                'train_r2': result['metrics'].get('train_r2', None),
                'test_r2': result['metrics'].get('test_r2', None),
                'train_rmse': result['metrics'].get('train_rmse', None),
                'test_rmse': result['metrics'].get('test_rmse', None)
            }
        
        return summary
    
    def save_results(self, filepath: str = None):
        """
        Save model results to file
        
        Args:
            filepath: Path to save file (optional)
        """
        if not self.model_results:
            logger.warning("No model results to save")
            return
        
        if filepath is None:
            from config import MODEL_RESULTS_PATH
            filepath = MODEL_RESULTS_PATH
        
        try:
            # Prepare results for saving
            results_data = []
            for model_key, result in self.model_results.items():
                row = {
                    'model_key': model_key,
                    'model_name': result['name']
                }
                row.update(result['metrics'])
                results_data.append(row)
            
            # Create DataFrame and save
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(filepath, index=False)
            
            logger.info(f"Model results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model results: {e}")
            raise
