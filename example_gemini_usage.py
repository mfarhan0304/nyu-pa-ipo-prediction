#!/usr/bin/env python3
"""
Example script demonstrating the optimized SEC Processor with Gemini API
======================================================================
This script shows how to use the new Gemini-powered SEC filing processor
with single API calls and file uploads for maximum efficiency
"""

import os
import pandas as pd
from sec_processor import SECFilingProcessor
from config import IPO_DATA_PATH

def main():
    """Main function to demonstrate optimized Gemini-powered SEC processing"""
    
    # Check if Gemini API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return
    
    print("🚀 Initializing Optimized Gemini-powered SEC Processor...")
    print("✨ Key Optimizations:")
    print("   • Single comprehensive API call per filing (vs. 7 separate calls)")
    print("   • File uploads instead of text in prompts")
    print("   • 7x reduction in API costs and latency")
    print("   • Better context and accuracy with full document analysis")
    print()
    
    try:
        # Initialize the processor
        processor = SECFilingProcessor()
        print("✅ SEC Processor initialized successfully with optimized Gemini API")
        
        # Load sample IPO data
        if IPO_DATA_PATH.exists():
            print(f"📊 Loading IPO data from {IPO_DATA_PATH}")
            ipo_df = pd.read_csv(IPO_DATA_PATH)
            print(f"✅ Loaded {len(ipo_df)} IPO records")
            
            # Process a small sample for demonstration
            sample_size = min(3, len(ipo_df))
            sample_df = ipo_df.head(sample_size)
            
            print(f"\n🔍 Processing {sample_size} sample filings with optimized Gemini API...")
            print("💡 This will make only 1 API call per filing instead of 7!")
            print("📁 Documents will be uploaded as files for better context...")
            print("⏱️  This may take a few minutes as we make comprehensive API calls...")
            
            # Process filings
            features_df = processor.process_all_filings(sample_df)
            
            if not features_df.empty:
                print(f"\n✅ Successfully processed {len(features_df)} filings!")
                print(f"💰 Total API calls made: {len(features_df)} (vs. {len(features_df) * 7} with old approach)")
                print(f"📉 API cost reduction: {((7-1)/7)*100:.1f}%")
                print("\n📋 Generated features:")
                
                # Show feature columns
                feature_columns = [col for col in features_df.columns if col.startswith('gemini_')]
                print(f"Gemini-generated features: {len(feature_columns)}")
                
                # Display sample features
                print("\n🔍 Sample Gemini features for first filing:")
                first_filing = features_df.iloc[0]
                for col in feature_columns[:10]:  # Show first 10 features
                    print(f"  {col}: {first_filing[col]}")
                
                # Show analysis coverage
                print("\n📊 Analysis Coverage (from single API call):")
                coverage_areas = {
                    'Sentiment': [col for col in feature_columns if 'sentiment' in col],
                    'Risk': [col for col in feature_columns if 'risk' in col],
                    'Business': [col for col in feature_columns if 'business' in col],
                    'Financial': [col for col in feature_columns if 'financial' in col],
                    'Market': [col for col in feature_columns if 'market' in col],
                    'Technology': [col for col in feature_columns if 'tech' in col],
                    'Compliance': [col for col in feature_columns if 'compliance' in col]
                }
                
                for area, features in coverage_areas.items():
                    print(f"  {area}: {len(features)} features")
                
                # Save features
                output_file = "optimized_gemini_features.csv"
                features_df.to_csv(output_file, index=False)
                print(f"\n💾 Features saved to {output_file}")
                
                # Performance summary
                print(f"\n🚀 Performance Summary:")
                print(f"   • API calls: {len(features_df)} (optimized)")
                print(f"   • Old approach would have used: {len(features_df) * 7} calls")
                print(f"   • Cost savings: {((7-1)/7)*100:.1f}%")
                print(f"   • Context quality: Full document analysis")
                print(f"   • Analysis depth: Comprehensive 7-area coverage")
                
            else:
                print("❌ No features were generated")
                
        else:
            print(f"❌ IPO data file not found: {IPO_DATA_PATH}")
            print("Please ensure you have the required data files")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure GEMINI_API_KEY is set correctly")
        print("2. Check your internet connection")
        print("3. Verify the Gemini API key is valid")
        print("4. Check if you have sufficient API quota")
        print("5. Verify SEC filing data exists in the expected directory")

if __name__ == "__main__":
    main()
