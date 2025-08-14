#!/usr/bin/env python3
"""
Standalone Visualization Generator for IPO Analysis
==================================================
Generate comprehensive visualizations from existing pipeline results
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from visualization_generator import VisualizationGenerator
from config import RESULTS_DIR, VISUALIZATIONS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_existing_results():
    """Load existing pipeline results for visualization"""
    results = {}
    
    try:
        # Load enhanced dataset
        enhanced_dataset_path = RESULTS_DIR / "enhanced_ipo_dataset.csv"
        if enhanced_dataset_path.exists():
            results['data'] = pd.read_csv(enhanced_dataset_path)
            logger.info(f"Loaded enhanced dataset: {results['data'].shape}")
        else:
            logger.warning("Enhanced dataset not found")
            return None
        
        # Load model results
        regression_results_path = RESULTS_DIR / "regression_results.csv"
        classification_results_path = RESULTS_DIR / "classification_results.csv"
        
        model_results = {}
        
        if regression_results_path.exists():
            reg_df = pd.read_csv(regression_results_path)
            reg_results = {}
            for _, row in reg_df.iterrows():
                reg_results[row['model_key']] = {
                    'name': row['name'],
                    'metrics': {
                        'train_r2': row.get('train_r2', 0),
                        'test_r2': row.get('test_r2', 0),
                        'train_mse': row.get('train_mse', 0),
                        'test_mse': row.get('test_mse', 0)
                    }
                }
            model_results['regression_results'] = reg_results
            logger.info(f"Loaded regression results: {len(reg_results)} models")
        
        if classification_results_path.exists():
            clf_df = pd.read_csv(classification_results_path)
            clf_results = {}
            for _, row in clf_df.iterrows():
                clf_results[row['model_key']] = {
                    'name': row['name'],
                    'metrics': {
                        'train_accuracy': row.get('train_accuracy', 0),
                        'test_accuracy': row.get('test_accuracy', 0)
                    }
                }
            model_results['classification_results'] = clf_results
            logger.info(f"Loaded classification results: {len(clf_results)} models")
        
        results['model_results'] = model_results
        
        # Load feature importance
        feature_importance_path = RESULTS_DIR / "feature_importance.csv"
        if feature_importance_path.exists():
            results['feature_importance'] = pd.read_csv(feature_importance_path)
            logger.info(f"Loaded feature importance: {results['feature_importance'].shape}")
        else:
            results['feature_importance'] = pd.DataFrame()
            logger.info("Feature importance not found, will skip feature analysis")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading existing results: {e}")
        return None

def main():
    """Main function to generate visualizations"""
    print("IPO Analysis - Visualization Generator")
    print("=" * 50)
    print("This script generates comprehensive visualizations from existing pipeline results")
    print("=" * 50)
    
    # Check if results exist
    if not RESULTS_DIR.exists():
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        print("Please run the main pipeline first to generate results")
        return
    
    # Load existing results
    print("\n📊 Loading existing pipeline results...")
    results = load_existing_results()
    
    if results is None:
        print("❌ Failed to load existing results")
        return
    
    # Initialize visualization generator
    print("\n🎨 Initializing visualization generator...")
    viz_generator = VisualizationGenerator()
    
    # Generate comprehensive report
    print("\n📈 Generating comprehensive visualizations...")
    try:
        visualization_files = viz_generator.generate_comprehensive_report(
            data=results['data'],
            model_results=results['model_results'],
            feature_importance=results['feature_importance']
        )
        
        print(f"\n✅ Successfully generated {len(visualization_files)} visualization files!")
        
        # Display generated files
        print("\n📁 Generated Visualizations:")
        for viz_type, filepath in visualization_files.items():
            print(f"  📊 {viz_type.replace('_', ' ').title()}: {filepath}")
        
        # Generate quick charts
        print("\n🚀 Generating quick analysis charts...")
        quick_charts = viz_generator.generate_quick_charts(results['data'])
        print(f"✅ Generated {len(quick_charts)} quick charts")
        
        print(f"\n📁 All visualizations saved to: {VISUALIZATIONS_DIR}")
        print("\n🎉 Visualization generation completed successfully!")
        
        # Open visualization directory
        try:
            import subprocess
            import platform
            
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(VISUALIZATIONS_DIR)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(VISUALIZATIONS_DIR)])
            else:  # Linux
                subprocess.run(["xdg-open", str(VISUALIZATIONS_DIR)])
                
            print("📂 Opened visualizations directory in file explorer")
        except:
            print("📂 Please manually open the visualizations directory")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        logger.error(f"Visualization generation failed: {e}")

if __name__ == "__main__":
    main()
