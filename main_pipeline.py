"""
Main Pipeline for IPO Analysis ML Pipeline
==========================================
Orchestrates the complete machine learning pipeline for IPO analysis
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# Import pipeline modules
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

# Import configuration
from config import (
    LOGGING_CONFIG, RESULTS_DIR, RANDOM_STATE,
    ENHANCED_DATASET_PATH, MODEL_RESULTS_PATH, 
    FEATURE_IMPORTANCE_PATH, PREDICTIONS_PATH
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IPOPipeline:
    """
    Main pipeline orchestrator for IPO analysis
    """
    
    def __init__(self):
        """Initialize the IPO pipeline"""
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # Pipeline state
        self.combined_data = None
        self.engineered_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info("IPO Pipeline initialized")
    
    def run_pipeline(self, max_filings: Optional[int] = None, 
                    enable_feature_selection: bool = True,
                    enable_pca: bool = False) -> bool:
        """
        Run the complete IPO analysis pipeline
        
        Args:
            max_filings: Maximum number of SEC filings to process
            enable_feature_selection: Whether to enable feature selection
            enable_pca: Whether to enable PCA dimensionality reduction
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING IPO ANALYSIS PIPELINE")
            logger.info("=" * 80)
            
            # Step 1: Load and merge data
            logger.info("\n" + "="*60)
            logger.info("STEP 1: DATA LOADING AND MERGING")
            logger.info("="*60)
            
            success = self._load_and_merge_data(max_filings)
            if not success:
                logger.error("Pipeline failed at data loading step")
                return False
            
            # Step 2: Feature engineering
            logger.info("\n" + "="*60)
            logger.info("STEP 2: FEATURE ENGINEERING")
            logger.info("="*60)
            
            success = self._engineer_features()
            if not success:
                logger.error("Pipeline failed at feature engineering step")
                return False
            
            # Step 3: Model training and evaluation
            logger.info("\n" + "="*60)
            logger.info("STEP 3: MODEL TRAINING AND EVALUATION")
            logger.info("="*60)
            
            success = self._train_and_evaluate_models(enable_feature_selection, enable_pca)
            if not success:
                logger.error("Pipeline failed at model training step")
                return False
            
            # Step 4: Save results and generate reports
            logger.info("\n" + "="*60)
            logger.info("STEP 4: SAVING RESULTS AND GENERATING REPORTS")
            logger.info("="*60)
            
            success = self._save_results_and_reports()
            if not success:
                logger.error("Pipeline failed at results saving step")
                return False
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            return False
    
    def _load_and_merge_data(self, max_filings: Optional[int] = None) -> bool:
        """
        Load and merge all data sources
        
        Args:
            max_filings: Maximum number of SEC filings to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load IPO data
            logger.info("Loading IPO data...")
            ipo_data = self.data_loader.load_ipo_data()
            if ipo_data.empty:
                logger.error("Failed to load IPO data")
                return False
            
            # Load market data
            logger.info("Loading market data...")
            vix_data, fedfunds_data = self.data_loader.load_market_data()
            
            # Load SEC filings (optional)
            if max_filings is not None:
                logger.info(f"Loading SEC filings (limited to {max_filings})...")
                sec_features = self.data_loader.load_sec_filings(max_filings=max_filings)
            else:
                logger.info("Loading all SEC filings...")
                sec_features = self.data_loader.load_sec_filings()
            
            # Merge all data
            logger.info("Merging all data sources...")
            self.combined_data = self.data_loader.merge_data()
            
            if self.combined_data.empty:
                logger.error("Failed to merge data")
                return False
            
            # Save combined dataset
            self.data_loader.save_combined_data()
            
            # Log data summary
            summary = self.data_loader.get_data_summary()
            logger.info(f"Data summary: {summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data loading and merging: {e}")
            return False
    
    def _engineer_features(self) -> bool:
        """
        Engineer features from the combined dataset
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.combined_data is None:
                logger.error("No combined data available for feature engineering")
                return False
            
            logger.info("Starting feature engineering...")
            
            # Check for Gemini API features (comma-separated strings)
            gemini_columns = self._identify_gemini_features()
            if gemini_columns:
                logger.info(f"Found {len(gemini_columns)} Gemini API feature columns: {gemini_columns}")
                logger.info("Applying specialized preprocessing for comma-separated strings...")
            
            # Engineer features (includes new preprocessing methods)
            self.engineered_features = self.feature_engineer.engineer_features(self.combined_data)
            
            if self.engineered_features.empty:
                logger.error("Feature engineering failed")
                return False
            
            logger.info(f"Feature engineering completed. Shape: {self.engineered_features.shape}")
            
            # Log feature summary
            feature_summary = self.feature_engineer.get_feature_summary()
            logger.info(f"Feature engineering summary: {feature_summary}")
            
            # Log preprocessing details
            self._log_preprocessing_details()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False
    
    def _identify_gemini_features(self) -> List[str]:
        """
        Identify columns that contain Gemini API features (comma-separated strings)
        
        Returns:
            List of column names containing Gemini API features
        """
        try:
            if self.combined_data is None:
                return []
            
            gemini_columns = []
            
            # Look for columns that start with 'gemini_' and contain comma-separated values
            for col in self.combined_data.columns:
                if col.startswith('gemini_') and self.combined_data[col].dtype == 'object':
                    # Check if column contains comma-separated values
                    sample_values = self.combined_data[col].dropna().head(100)
                    if len(sample_values) > 0:
                        has_commas = sample_values.str.contains(',').any()
                        if has_commas:
                            gemini_columns.append(col)
            
            return gemini_columns
            
        except Exception as e:
            logger.warning(f"Error identifying Gemini features: {e}")
            return []
    
    def _log_preprocessing_details(self):
        """Log detailed information about preprocessing steps"""
        try:
            if self.engineered_features is None:
                return
            
            logger.info("\n" + "="*50)
            logger.info("PREPROCESSING DETAILS")
            logger.info("="*50)
            
            # Count different types of features
            feature_counts = {}
            
            for col in self.engineered_features.columns:
                if '_present' in col:
                    feature_counts['keyword_binary'] = feature_counts.get('keyword_binary', 0) + 1
                elif '_tfidf_' in col:
                    feature_counts['tfidf'] = feature_counts.get('tfidf', 0) + 1
                elif '_item_count' in col or '_string_length' in col or '_has_multiple' in col:
                    feature_counts['count_features'] = feature_counts.get('count_features', 0) + 1
                elif '_encoded' in col:
                    feature_counts['categorical_encoded'] = feature_counts.get('categorical_encoded', 0) + 1
                else:
                    feature_counts['other'] = feature_counts.get('other', 0) + 1
            
            logger.info("Feature type breakdown:")
            for feature_type, count in feature_counts.items():
                logger.info(f"  {feature_type}: {count}")
            
            # Log preprocessing method used
            if hasattr(self.feature_engineer, 'tfidf_vectorizers'):
                tfidf_cols = list(self.feature_engineer.tfidf_vectorizers.keys())
                logger.info(f"TF-IDF preprocessing applied to: {tfidf_cols}")
            
            logger.info("="*50)
            
        except Exception as e:
            logger.warning(f"Error logging preprocessing details: {e}")
    
    def _train_and_evaluate_models(self, enable_feature_selection: bool = True, 
                                 enable_pca: bool = False) -> bool:
        """
        Train and evaluate machine learning models
        
        Args:
            enable_feature_selection: Whether to enable feature selection
            enable_pca: Whether to enable PCA dimensionality reduction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.engineered_features is None:
                logger.error("No engineered features available for modeling")
                return False
            
            # Prepare target variable
            if 'first_day_return' not in self.combined_data.columns:
                logger.error("Target variable 'first_day_return' not found")
                return False
            
            y = self.combined_data['first_day_return']
            X = self.engineered_features
            
            # Prepare data for training
            logger.info("Preparing data for training...")
            X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(X, y)
            
            # Store for later use
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            
            # Scale features
            logger.info("Scaling features...")
            X_train_scaled, X_test_scaled = self.feature_engineer.scale_features(X_train, X_test)
            
            # Apply feature selection if enabled
            if enable_feature_selection:
                logger.info("Applying feature selection...")
                X_train_selected = self.feature_engineer.apply_feature_selection(X_train_scaled, y_train)
                X_test_selected = self.feature_engineer.apply_feature_selection(X_test_scaled, y_test)
                
                X_train_final = X_train_selected
                X_test_final = X_test_selected
            else:
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
            
            # Apply PCA if enabled
            if enable_pca:
                logger.info("Applying PCA dimensionality reduction...")
                # Fit PCA on training data only
                X_train_final = self.feature_engineer.fit_pca(X_train_final)
                # Transform test data using fitted PCA
                X_test_final = self.feature_engineer.transform_pca(X_test_final)

            self.X_test = X_test_final
            
            # Train models
            logger.info("Training models...")
            training_results = self.model_trainer.train_models(
                X_train_final, y_train, X_test_final, y_test
            )
            
            if not training_results:
                logger.error("Model training failed")
                return False
            
            # Perform cross-validation
            logger.info("Performing cross-validation...")
            cv_results = self.model_trainer.cross_validate_models(X_train_final, y_train)
                        
            # Get feature importance
            feature_importance = self.model_trainer.get_feature_importance()
            
            logger.info("Model training and evaluation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in model training and evaluation: {e}")
            return False
    
    def _save_results_and_reports(self) -> bool:
        """
        Save results and generate reports
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Saving results and generating reports...")
            
            # Save model results
            self.model_trainer.save_results()
            
            # Save feature importance
            feature_importance = self.model_trainer.get_feature_importance()
            if not feature_importance.empty:
                feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
                logger.info(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")
            
            # Save predictions
            if self.X_test is not None and self.y_test is not None:
                predictions = self.model_trainer.make_predictions(self.X_test)
                predictions_df = pd.DataFrame({
                    'actual': self.y_test,
                    'predicted': predictions
                })
                predictions_df.to_csv(PREDICTIONS_PATH, index=False)
                logger.info(f"Predictions saved to {PREDICTIONS_PATH}")
            
            # Generate pipeline summary
            self._generate_pipeline_summary()
            
            logger.info("Results and reports saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results and reports: {e}")
            return False
    
    def _generate_pipeline_summary(self):
        """Generate and log pipeline summary"""
        try:
            logger.info("\n" + "="*60)
            logger.info("PIPELINE SUMMARY")
            logger.info("="*60)
            
            # Data summary
            if self.combined_data is not None:
                logger.info(f"Combined dataset: {self.combined_data.shape}")
            
            # Feature summary
            if self.engineered_features is not None:
                logger.info(f"Engineered features: {self.engineered_features.shape}")
            
            # Model summary
            model_summary = self.model_trainer.get_model_summary()
            logger.info(f"Models trained: {model_summary['trained_models']}")
            logger.info(f"Best model: {model_summary['best_model']}")
            
            # Feature engineering summary
            feature_summary = self.feature_engineer.get_feature_summary()
            logger.info(f"Total features: {feature_summary['total_features']}")
            logger.info(f"Selected features: {feature_summary['selected_features']}")
            
            # Preprocessing summary
            preprocessing_summary = self.get_preprocessing_summary()
            if preprocessing_summary:
                logger.info(f"Preprocessing methods: {', '.join(preprocessing_summary.get('preprocessing_methods', []))}")
                logger.info("Feature type breakdown:")
                for feature_type, count in preprocessing_summary.get('feature_types', {}).items():
                    logger.info(f"  {feature_type}: {count}")
            
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error generating pipeline summary: {e}")
    
    def get_pipeline_results(self) -> Dict:
        """
        Get comprehensive pipeline results
        
        Returns:
            Dictionary with pipeline results
        """
        results = {
            'data_summary': self.data_loader.get_data_summary() if self.data_loader else {},
            'feature_summary': self.feature_engineer.get_feature_summary() if self.feature_engineer else {},
            'model_summary': self.model_trainer.get_model_summary() if self.model_trainer else {},
            'pipeline_status': 'completed' if self.combined_data is not None else 'not_started'
        }
        
        return results
    
    def preprocess_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for prediction using fitted transformers
        
        Args:
            new_data: New DataFrame for prediction
            
        Returns:
            DataFrame preprocessed for prediction
        """
        try:
            if self.feature_engineer is None:
                logger.error("Feature engineer not initialized")
                return new_data
            
            logger.info("Preprocessing new data for prediction...")
            
            # Check for Gemini API features
            gemini_columns = self._identify_gemini_features_in_data(new_data)
            if gemini_columns:
                logger.info(f"Found {len(gemini_columns)} Gemini API feature columns in new data")
            
            # Apply preprocessing using fitted transformers
            preprocessed_data = self.feature_engineer.preprocess_for_prediction(new_data)
            
            logger.info(f"Preprocessing completed. Shape: {preprocessed_data.shape}")
            return preprocessed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing new data: {e}")
            return new_data
    
    def _identify_gemini_features_in_data(self, data: pd.DataFrame) -> List[str]:
        """
        Identify Gemini API features in a given DataFrame
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of column names containing Gemini API features
        """
        try:
            gemini_columns = []
            
            # Look for columns that start with 'gemini_' and contain comma-separated values
            for col in data.columns:
                if col.startswith('gemini_') and data[col].dtype == 'object':
                    # Check if column contains comma-separated values
                    sample_values = data[col].dropna().head(100)
                    if len(sample_values) > 0:
                        has_commas = sample_values.str.contains(',').any()
                        if has_commas:
                            gemini_columns.append(col)
            
            return gemini_columns
            
        except Exception as e:
            logger.warning(f"Error identifying Gemini features in data: {e}")
            return []
    
    def predict_new_data(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data using the trained model
        
        Args:
            new_data: New DataFrame for prediction
            
        Returns:
            Array of predictions
        """
        try:
            if self.model_trainer is None:
                logger.error("Model trainer not initialized")
                return np.array([])
            
            if not hasattr(self.model_trainer, 'best_model') or self.model_trainer.best_model is None:
                logger.error("No trained model available for prediction")
                return np.array([])
            
            logger.info("Making predictions on new data...")
            
            # Preprocess new data
            preprocessed_data = self.preprocess_new_data(new_data)
            
            # Make predictions
            predictions = self.model_trainer.make_predictions(preprocessed_data)
            
            logger.info(f"Predictions completed. Shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get detailed summary of preprocessing steps applied
        
        Returns:
            Dictionary with preprocessing summary
        """
        try:
            if self.engineered_features is None:
                return {}
            
            summary = {
                'total_features': len(self.engineered_features.columns),
                'feature_types': {},
                'preprocessing_methods': []
            }
            
            # Count feature types
            for col in self.engineered_features.columns:
                if '_present' in col:
                    summary['feature_types']['keyword_binary'] = summary['feature_types'].get('keyword_binary', 0) + 1
                elif '_tfidf_' in col:
                    summary['feature_types']['tfidf'] = summary['feature_types'].get('tfidf', 0) + 1
                elif '_item_count' in col or '_string_length' in col or '_has_multiple' in col:
                    summary['feature_types']['count_features'] = summary['feature_types'].get('count_features', 0) + 1
                elif '_encoded' in col:
                    summary['feature_types']['categorical_encoded'] = summary['feature_types'].get('categorical_encoded', 0) + 1
                else:
                    summary['feature_types']['other'] = summary['feature_types'].get('other', 0) + 1
            
            # Get preprocessing methods used
            if hasattr(self.feature_engineer, 'tfidf_vectorizers'):
                summary['preprocessing_methods'].append('TF-IDF Vectorization')
            
            if any('_present' in col for col in self.engineered_features.columns):
                summary['preprocessing_methods'].append('Keyword Extraction')
            
            if any('_item_count' in col for col in self.engineered_features.columns):
                summary['preprocessing_methods'].append('Count Features')
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting preprocessing summary: {e}")
            return {}


def main():
    """Main execution function"""
    print("IPO Analysis ML Pipeline")
    print("=" * 50)
    print("This pipeline combines:")
    print("1. Traditional IPO features (CSV data)")
    print("2. Advanced NLP analysis of SEC filings (Gemini API)")
    print("3. Enhanced feature engineering with specialized preprocessing")
    print("4. Comprehensive machine learning modeling")
    print("=" * 50)
    print("\n🆕 NEW: Advanced preprocessing for comma-separated strings:")
    print("   • Keyword extraction (binary features)")
    print("   • TF-IDF vectorization")
    print("   • Count-based features")
    print("   • Automatic Gemini API feature detection")
    print("=" * 50)
    
    # Get user input
    max_filings = input("Enter maximum number of filings to process (or press Enter for all): ").strip()
    max_filings = int(max_filings) if max_filings.isdigit() else None
    
    enable_feature_selection = input("Enable feature selection? (y/n, default: y): ").strip().lower()
    enable_feature_selection = enable_feature_selection != 'n'
    
    enable_pca = input("Enable PCA dimensionality reduction? (y/n, default: n): ").strip().lower()
    enable_pca = enable_pca == 'y'
    
    if max_filings:
        print(f"Processing limited to {max_filings} filings")
    
    print(f"Feature selection: {'Enabled' if enable_feature_selection else 'Disabled'}")
    print(f"PCA: {'Enabled' if enable_pca else 'Disabled'}")
    
    # Create and run pipeline
    pipeline = IPOPipeline()
    success = pipeline.run_pipeline(
        max_filings=max_filings,
        enable_feature_selection=enable_feature_selection,
        enable_pca=enable_pca
    )
    
    if success:
        print(f"\n🎉 Pipeline completed successfully!")
        
        # Get and display results
        results = pipeline.get_pipeline_results()
        
        print(f"\n📊 Pipeline Results:")
        print(f"- Data records: {results['data_summary'].get('combined_records', 0)}")
        print(f"- Features: {results['feature_summary'].get('total_features', 0)}")
        print(f"- Models trained: {results['model_summary'].get('trained_models', 0)}")
        print(f"- Best model: {results['model_summary'].get('best_model', 'N/A')}")
        
        # Show preprocessing summary
        preprocessing_summary = pipeline.get_preprocessing_summary()
        if preprocessing_summary:
            print(f"\n🔧 Preprocessing Applied:")
            print(f"- Methods: {', '.join(preprocessing_summary.get('preprocessing_methods', []))}")
            print(f"- Feature types:")
            for feature_type, count in preprocessing_summary.get('feature_types', {}).items():
                print(f"  • {feature_type}: {count}")
        
        print(f"\n📁 Output files created in 'results/' directory:")
        print("- enhanced_ipo_dataset.csv")
        print("- model_results.csv")
        print("- feature_importance.csv")
        print("- predictions.csv")        
    else:
        print("\n❌ Pipeline failed. Check logs for details.")


if __name__ == "__main__":
    main()
