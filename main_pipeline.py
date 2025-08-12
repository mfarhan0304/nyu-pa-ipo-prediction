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
            
            # Engineer features
            self.engineered_features = self.feature_engineer.engineer_features(self.combined_data)
            
            if self.engineered_features.empty:
                logger.error("Feature engineering failed")
                return False
            
            logger.info(f"Feature engineering completed. Shape: {self.engineered_features.shape}")
            
            # Log feature summary
            feature_summary = self.feature_engineer.get_feature_summary()
            logger.info(f"Feature engineering summary: {feature_summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False
    
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


def main():
    """Main execution function"""
    print("IPO Analysis ML Pipeline")
    print("=" * 50)
    print("This pipeline combines:")
    print("1. Traditional IPO features (CSV data)")
    print("2. Advanced NLP analysis of SEC filings")
    print("3. Enhanced feature engineering")
    print("4. Comprehensive machine learning modeling")
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
        
        print(f"\n📁 Output files created in 'results/' directory:")
        print("- enhanced_ipo_dataset.csv")
        print("- model_results.csv")
        print("- feature_importance.csv")
        print("- predictions.csv")        
    else:
        print("\n❌ Pipeline failed. Check logs for details.")


if __name__ == "__main__":
    main()
