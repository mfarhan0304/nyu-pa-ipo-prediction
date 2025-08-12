#!/usr/bin/env python3
"""
Example script demonstrating preprocessing of comma-separated strings
================================================================
Shows how to handle list values from Gemini API during training and prediction
"""

import pandas as pd
import numpy as np
from feature_engineer import FeatureEngineer

def create_sample_data():
    """Create sample data with comma-separated strings (like from Gemini API)"""
    
    print("📊 Creating sample data with comma-separated strings...")
    
    sample_data = {
        'company_name': ['Apple Inc', 'Google LLC', 'Microsoft Corp', 'Tesla Inc', 'Amazon.com'],
        'filing_type': ['S1', 'S1', 'S1', 'S1', 'S1'],
        'cik': ['0000320193', '0001652044', '0000789019', '0001318605', '0001018724'],
        
        # These would come from Gemini API analysis
        'gemini_risk_primary_risk_categories': [
            'market volatility, technology disruption',
            'regulatory compliance, competition',
            'cybersecurity, privacy concerns',
            'supply chain, regulatory',
            'market competition, regulatory'
        ],
        'gemini_business_competitive_advantages': [
            'brand recognition, ecosystem lock-in',
            'search dominance, AI leadership',
            'enterprise software, cloud platform',
            'electric vehicle technology, brand',
            'e-commerce platform, cloud services'
        ],
        'gemini_tech_technology_sophistication': [
            'cutting_edge',
            'advanced',
            'advanced',
            'cutting_edge',
            'advanced'
        ],
        
        # Traditional features
        'price': [150.0, 2800.0, 300.0, 800.0, 3500.0],
        'shares': [1000000, 500000, 2000000, 800000, 1500000],
        'employees': [154000, 135301, 181000, 127855, 1608000]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Created sample data with shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df

def demonstrate_preprocessing_methods():
    """Demonstrate different preprocessing methods for comma-separated strings"""
    
    print("\n🔄 Demonstrating preprocessing methods...\n")
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    print("🔍 Original data sample:")
    print(df[['gemini_risk_primary_risk_categories', 'gemini_business_competitive_advantages']].head())
    
    # Method 1: Keyword extraction (binary features)
    print("\n1️⃣ Keyword Extraction Method:")
    print("   Creates binary features for each unique keyword")
    
    keyword_df = engineer._extract_keywords_from_strings(df.copy(), ['gemini_risk_primary_risk_categories'])
    keyword_cols = [col for col in keyword_df.columns if 'primary_risk_categories' in col and 'present' in col]
    print(f"   Created {len(keyword_cols)} binary features:")
    for col in keyword_cols[:5]:  # Show first 5
        print(f"     {col}")
    
    # Method 2: TF-IDF features
    print("\n2️⃣ TF-IDF Method:")
    print("   Creates numerical features based on term frequency")
    
    tfidf_df = engineer._create_tfidf_features(df.copy(), ['gemini_risk_primary_risk_categories'])
    tfidf_cols = [col for col in tfidf_df.columns if 'tfidf' in col]
    print(f"   Created {len(tfidf_cols)} TF-IDF features:")
    for col in tfidf_cols[:5]:  # Show first 5
        print(f"     {col}")
    
    # Method 3: Count features
    print("\n3️⃣ Count Method:")
    print("   Creates numerical features based on counts and lengths")
    
    count_df = engineer._create_count_features(df.copy(), ['gemini_risk_primary_risk_categories'])
    count_cols = [col for col in count_df.columns if 'item_count' in col or 'string_length' in col or 'has_multiple' in col]
    print(f"   Created {len(count_cols)} count features:")
    for col in count_cols:
        print(f"     {col}")
    
    return df, engineer

def demonstrate_training_pipeline():
    """Demonstrate the complete training pipeline"""
    
    print("\n🚀 Demonstrating Complete Training Pipeline...\n")
    
    # Create sample data
    df, engineer = demonstrate_preprocessing_methods()
    
    print("🔧 Running complete feature engineering pipeline...")
    
    try:
        # This will apply all preprocessing methods
        engineered_df = engineer.engineer_features(df.copy())
        
        print(f"✅ Feature engineering completed!")
        print(f"📊 Original shape: {df.shape}")
        print(f"📊 Engineered shape: {engineered_df.shape}")
        print(f"📈 Features added: {engineered_df.shape[1] - df.shape[1]}")
        
        # Show some of the new features
        new_features = [col for col in engineered_df.columns if col not in df.columns]
        print(f"\n🆕 New features created: {len(new_features)}")
        print("   Sample new features:")
        for feature in new_features[:10]:  # Show first 10
            print(f"     {feature}")
        
        return engineer, engineered_df
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return None, None

def demonstrate_prediction_pipeline(engineer, training_df):
    """Demonstrate prediction preprocessing with new data"""
    
    print("\n🔮 Demonstrating Prediction Pipeline...\n")
    
    # Create new data for prediction (simulating new filings)
    new_data = {
        'company_name': ['Netflix Inc', 'Zoom Video'],
        'filing_type': ['S1', 'S1'],
        'cik': ['0001065280', '0001585521'],
        'gemini_risk_primary_risk_categories': [
            'content competition, streaming market',
            'cybersecurity, remote work trends'
        ],
        'gemini_business_competitive_advantages': [
            'content library, recommendation algorithm',
            'video conferencing, ease of use'
        ],
        'gemini_tech_technology_sophistication': [
            'advanced',
            'standard'
        ],
        'price': [450.0, 120.0],
        'shares': [600000, 400000],
        'employees': [12500, 4500]
    }
    
    new_df = pd.DataFrame(new_data)
    
    print("📊 New data for prediction:")
    print(new_df[['gemini_risk_primary_risk_categories', 'gemini_business_competitive_advantages']])
    
    print("\n🔧 Preprocessing new data for prediction...")
    
    try:
        # This uses the fitted transformers from training
        preprocessed_df = engineer.preprocess_for_prediction(new_df.copy())
        
        print(f"✅ Prediction preprocessing completed!")
        print(f"📊 New data shape: {new_df.shape}")
        print(f"📊 Preprocessed shape: {preprocessed_df.shape}")
        
        # Ensure same features as training
        if hasattr(engineer, 'feature_names'):
            print(f"🎯 Training features: {len(engineer.feature_names)}")
            print(f"🎯 Prediction features: {len(preprocessed_df.columns)}")
            
            # Check if we have the same features
            missing_features = set(engineer.feature_names) - set(preprocessed_df.columns)
            extra_features = set(preprocessed_df.columns) - set(engineer.feature_names)
            
            if missing_features:
                print(f"⚠️  Missing features: {len(missing_features)}")
            if extra_features:
                print(f"⚠️  Extra features: {len(extra_features)}")
            
            if not missing_features and not extra_features:
                print("✅ Feature alignment perfect!")
        
        return preprocessed_df
        
    except Exception as e:
        print(f"❌ Error in prediction preprocessing: {e}")
        return None

def main():
    """Main demonstration function"""
    
    print("🎯 Comma-Separated String Preprocessing Demonstration")
    print("=" * 60)
    print("This shows how to handle list values from Gemini API during ML training and prediction\n")
    
    try:
        # Demonstrate training pipeline
        engineer, training_df = demonstrate_training_pipeline()
        
        if engineer is not None and training_df is not None:
            # Demonstrate prediction pipeline
            prediction_df = demonstrate_prediction_pipeline(engineer, training_df)
            
            if prediction_df is not None:
                print("\n🎉 Successfully demonstrated both training and prediction pipelines!")
                print("\n📋 Summary of preprocessing methods:")
                print("   1. Keyword Extraction: Binary features for each unique term")
                print("   2. TF-IDF: Numerical features based on term frequency")
                print("   3. Count Features: Item count, string length, multiplicity")
                print("   4. Categorical Encoding: Label encoding for remaining categorical features")
                print("\n💡 Benefits:")
                print("   • Preserves all information from Gemini API lists")
                print("   • Creates ML-compatible numerical features")
                print("   • Handles both training and prediction consistently")
                print("   • Automatic feature alignment between training and prediction")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all required packages are installed")
        print("2. Check that the feature_engineer.py file is in the same directory")
        print("3. Verify the sample data structure")

if __name__ == "__main__":
    main()
