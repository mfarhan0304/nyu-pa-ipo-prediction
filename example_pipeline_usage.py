#!/usr/bin/env python3
"""
Example script demonstrating the updated main pipeline
===================================================
Shows how to use the new preprocessing capabilities for comma-separated strings
"""

import pandas as pd
import numpy as np
from main_pipeline import IPOPipeline

def create_sample_gemini_data():
    """Create sample data with Gemini API features for testing"""
    
    print("📊 Creating sample data with Gemini API features...")
    
    sample_data = {
        'company_name': ['TechCorp Inc', 'BioTech LLC', 'GreenEnergy Corp'],
        'filing_type': ['S1', 'S1', 'S1'],
        'cik': ['0000123456', '0000654321', '0000789012'],
        'first_day_return': [15.5, -2.3, 8.7],  # Target variable
        
        # Gemini API features (comma-separated strings)
        'gemini_risk_primary_risk_categories': [
            'market volatility, technology disruption, regulatory compliance',
            'clinical trial failure, FDA approval, patent expiration',
            'energy policy changes, commodity price volatility, regulatory compliance'
        ],
        'gemini_business_competitive_advantages': [
            'AI technology, network effects, brand recognition',
            'patented technology, research expertise, regulatory relationships',
            'renewable technology, government contracts, cost leadership'
        ],
        'gemini_tech_technology_sophistication': [
            'cutting_edge',
            'advanced',
            'standard'
        ],
        
        # Traditional features
        'price': [25.0, 45.0, 18.0],
        'shares': [2000000, 1500000, 3000000],
        'employees': [500, 1200, 800]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Created sample data with shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df

def demonstrate_pipeline_with_preprocessing():
    """Demonstrate the pipeline with new preprocessing capabilities"""
    
    print("\n🚀 Demonstrating Updated Pipeline with Preprocessing...\n")
    
    # Create sample data
    sample_data = create_sample_gemini_data()
    
    # Initialize pipeline
    pipeline = IPOPipeline()
    
    # Simulate loading data (in real usage, this would come from data_loader)
    pipeline.combined_data = sample_data
    
    print("🔍 Analyzing Gemini API features...")
    
    # Check for Gemini features
    gemini_columns = pipeline._identify_gemini_features()
    if gemini_columns:
        print(f"✅ Found {len(gemini_columns)} Gemini API feature columns:")
        for col in gemini_columns:
            print(f"   • {col}")
            # Show sample values
            sample_values = sample_data[col].head(2)
            for i, val in enumerate(sample_values):
                print(f"     Sample {i+1}: {val}")
    
    print("\n🔧 Running feature engineering with preprocessing...")
    
    try:
        # Run feature engineering (includes new preprocessing)
        success = pipeline._engineer_features()
        
        if success:
            print("✅ Feature engineering completed successfully!")
            
            # Get preprocessing summary
            preprocessing_summary = pipeline.get_preprocessing_summary()
            if preprocessing_summary:
                print(f"\n📊 Preprocessing Summary:")
                print(f"- Total features: {preprocessing_summary['total_features']}")
                print(f"- Methods applied: {', '.join(preprocessing_summary['preprocessing_methods'])}")
                print(f"- Feature breakdown:")
                for feature_type, count in preprocessing_summary['feature_types'].items():
                    print(f"  • {feature_type}: {count}")
            
            # Show some of the new features
            if pipeline.engineered_features is not None:
                new_features = [col for col in pipeline.engineered_features.columns if col not in sample_data.columns]
                print(f"\n🆕 New features created: {len(new_features)}")
                print("   Sample new features:")
                for feature in new_features[:10]:  # Show first 10
                    print(f"     {feature}")
            
            return pipeline
            
        else:
            print("❌ Feature engineering failed")
            return None
            
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return None

def demonstrate_prediction_pipeline(pipeline):
    """Demonstrate prediction capabilities with new data"""
    
    print("\n🔮 Demonstrating Prediction Pipeline...\n")
    
    # Create new data for prediction
    new_data = {
        'company_name': ['NewTech Inc', 'Innovation Corp'],
        'filing_type': ['S1', 'S1'],
        'cik': ['0000999999', '0000888888'],
        'gemini_risk_primary_risk_categories': [
            'cybersecurity threats, market competition',
            'supply chain disruption, regulatory changes'
        ],
        'gemini_business_competitive_advantages': [
            'blockchain technology, first-mover advantage',
            'AI algorithms, data network effects'
        ],
        'gemini_tech_technology_sophistication': [
            'cutting_edge',
            'advanced'
        ],
        'price': [35.0, 28.0],
        'shares': [1800000, 2200000],
        'employees': [300, 450]
    }
    
    new_df = pd.DataFrame(new_data)
    
    print("📊 New data for prediction:")
    print(new_df[['gemini_risk_primary_risk_categories', 'gemini_business_competitive_advantages']])
    
    print("\n🔧 Preprocessing new data...")
    
    try:
        # Preprocess new data
        preprocessed_data = pipeline.preprocess_new_data(new_df)
        
        print(f"✅ Preprocessing completed!")
        print(f"📊 Original shape: {new_df.shape}")
        print(f"📊 Preprocessed shape: {preprocessed_data.shape}")
        
        # Show some of the preprocessed features
        if not preprocessed_data.empty:
            print(f"\n🔍 Preprocessed features sample:")
            sample_cols = preprocessed_data.columns[:10]  # Show first 10
            for col in sample_cols:
                print(f"   {col}")
        
        return preprocessed_data
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None

def main():
    """Main demonstration function"""
    
    print("🎯 Updated Main Pipeline with Preprocessing Demonstration")
    print("=" * 70)
    print("This demonstrates the new preprocessing capabilities for comma-separated strings\n")
    
    try:
        # Demonstrate feature engineering with preprocessing
        pipeline = demonstrate_pipeline_with_preprocessing()
        
        if pipeline is not None:
            # Demonstrate prediction pipeline
            preprocessed_data = demonstrate_prediction_pipeline(pipeline)
            
            if preprocessed_data is not None:
                print("\n🎉 Successfully demonstrated both pipelines!")
                print("\n📋 Summary of new capabilities:")
                print("   1. Automatic Gemini API feature detection")
                print("   2. Specialized preprocessing for comma-separated strings")
                print("   3. Keyword extraction (binary features)")
                print("   4. TF-IDF vectorization")
                print("   5. Count-based features")
                print("   6. Seamless integration with existing pipeline")
                print("\n💡 Benefits:")
                print("   • Preserves all information from Gemini API")
                print("   • Creates ML-compatible numerical features")
                print("   • Handles both training and prediction consistently")
                print("   • Automatic feature alignment")
                print("   • Detailed logging and monitoring")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all required packages are installed")
        print("2. Check that main_pipeline.py and feature_engineer.py are in the same directory")
        print("3. Verify the sample data structure")

if __name__ == "__main__":
    main()
