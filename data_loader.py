"""
Data Loader Module for IPO Analysis ML Pipeline
===============================================
Handles loading and preprocessing of all data sources including CSV files and SEC filings
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from config import (
    IPO_DATA_PATH, VIX_DATA_PATH, FEDFUNDS_DATA_PATH, SEC_FILINGS_DIR,
    RANDOM_STATE, NLP_CONFIG
)

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Centralized data loader for all data sources in the IPO analysis pipeline
    """
    
    def __init__(self):
        """Initialize the data loader"""
        self.ipo_data = None
        self.vix_data = None
        self.fedfunds_data = None
        self.sec_features = None
        self.combined_data = None
        
        logger.info("DataLoader initialized")
    
    def load_ipo_data(self) -> pd.DataFrame:
        """
        Load and preprocess IPO data from CSV
        
        Returns:
            Preprocessed IPO DataFrame
        """
        try:
            logger.info(f"Loading IPO data from {IPO_DATA_PATH}")
            
            if not IPO_DATA_PATH.exists():
                raise FileNotFoundError(f"IPO data file not found: {IPO_DATA_PATH}")
            
            # Load data
            df = pd.read_csv(IPO_DATA_PATH)
            logger.info(f"Loaded {len(df)} IPO records")
            
            # Basic preprocessing
            df = self._preprocess_ipo_data(df)
            
            self.ipo_data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading IPO data: {e}")
            raise
    
    def load_market_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load VIX and Federal Funds Rate data
        
        Returns:
            Tuple of (vix_data, fedfunds_data)
        """
        try:
            # Load VIX data
            if VIX_DATA_PATH.exists():
                self.vix_data = pd.read_csv(VIX_DATA_PATH)
                self.vix_data['Date'] = pd.to_datetime(self.vix_data['Date'])
                logger.info(f"Loaded {len(self.vix_data)} VIX records")
            else:
                logger.warning(f"VIX data file not found: {VIX_DATA_PATH}")
                self.vix_data = pd.DataFrame()
            
            # Load Federal Funds Rate data
            if FEDFUNDS_DATA_PATH.exists():
                self.fedfunds_data = pd.read_csv(FEDFUNDS_DATA_PATH)
                self.fedfunds_data['observation_date'] = pd.to_datetime(self.fedfunds_data['observation_date'])
                logger.info(f"Loaded {len(self.fedfunds_data)} Fed Funds Rate records")
            else:
                logger.warning(f"Fed Funds Rate data file not found: {FEDFUNDS_DATA_PATH}")
                self.fedfunds_data = pd.DataFrame()
            
            return self.vix_data, self.fedfunds_data
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            raise
    
    def load_sec_filings(self, max_filings: Optional[int] = None) -> pd.DataFrame:
        """
        Load and process SEC filing data
        
        Args:
            max_filings: Maximum number of filings to process
            
        Returns:
            DataFrame with SEC filing features
        """
        try:
            logger.info("Loading SEC filing data...")
            
            if not SEC_FILINGS_DIR.exists():
                logger.warning(f"SEC filings directory not found: {SEC_FILINGS_DIR}")
                return pd.DataFrame()
            
            # Process SEC filings
            from sec_processor import SECFilingProcessor
            processor = SECFilingProcessor()
            
            if self.ipo_data is None:
                self.load_ipo_data()
            
            # Limit processing if specified
            if max_filings:
                ipo_subset = self.ipo_data.head(max_filings)
                logger.info(f"Processing limited to {max_filings} filings")
            else:
                ipo_subset = self.ipo_data
            
            # Process filings
            self.sec_features = processor.process_all_filings(ipo_subset)
            
            if not self.sec_features.empty:
                logger.info(f"Successfully processed {len(self.sec_features)} SEC filings")
                # Save features
                processor.save_features(self.sec_features, "sec_filing_features.csv")
            
            return self.sec_features
            
        except Exception as e:
            logger.error(f"Error loading SEC filings: {e}")
            raise
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merge all data sources into a combined dataset
        
        Returns:
            Combined DataFrame with all features
        """
        try:
            logger.info("Merging all data sources...")
            
            if self.ipo_data is None:
                self.load_ipo_data()
            
            if self.vix_data is None or self.fedfunds_data is None:
                self.load_market_data()
            
            # Start with IPO data
            combined = self.ipo_data.copy()
            
            # Merge market data
            combined = self._merge_market_data(combined)
            
            # Merge SEC features if available
            if self.sec_features is not None and not self.sec_features.empty:
                combined = self._merge_sec_features(combined)
            
            # Final preprocessing
            combined = self._final_preprocessing(combined)
            
            self.combined_data = combined
            logger.info(f"Combined dataset shape: {combined.shape}")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            raise
    
    def _preprocess_ipo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess IPO data
        
        Args:
            df: Raw IPO DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Clean and validate data
            df = df.dropna(subset=['filing', 'CIK'])
            
            # Ensure CIK is properly formatted
            df['CIK'] = df['CIK'].astype(str).str.zfill(10)
            
            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Clean numeric columns
            numeric_columns = ['Price', 'Shares', 'Employees', 'Total Offering Expense']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean text columns
            text_columns = ['Symbol', 'Company Name', 'filing']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            
            logger.info("IPO data preprocessing completed")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing IPO data: {e}")
            raise
    
    def _merge_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge market data (VIX and Fed Funds Rate) with IPO data
        
        Args:
            df: IPO DataFrame
            
        Returns:
            DataFrame with merged market data
        """
        try:
            if self.vix_data.empty and self.fedfunds_data.empty:
                logger.warning("No market data available for merging")
                return df
            
            # Merge VIX data
            if not self.vix_data.empty:
                df = self._get_closest_market_data(df, self.vix_data, 'VIX', 'VIX')
            
            # Merge Fed Funds Rate data
            if not self.fedfunds_data.empty:
                df = self._get_closest_market_data(df, self.fedfunds_data, 'FEDFUNDS', 'FEDFUNDS')
            
            logger.info("Market data merging completed")
            return df
            
        except Exception as e:
            logger.error(f"Error merging market data: {e}")
            raise
    
    def _get_closest_market_data(self, ipo_df: pd.DataFrame, market_df: pd.DataFrame, 
                                value_col: str, new_col_name: str) -> pd.DataFrame:
        """
        Get closest market data for each IPO date
        
        Args:
            ipo_df: IPO DataFrame
            market_df: Market data DataFrame
            value_col: Column name in market data
            new_col_name: New column name in IPO data
            
        Returns:
            DataFrame with merged market data
        """
        try:
            if 'Date' not in ipo_df.columns or 'Date' not in market_df.columns:
                logger.warning(f"Date column not found in one of the DataFrames")
                return ipo_df
            
            # Find closest market data for each IPO
            closest_values = []
            for _, ipo_row in ipo_df.iterrows():
                ipo_date = ipo_row['Date']
                if pd.isna(ipo_date):
                    closest_values.append(np.nan)
                    continue
                
                # Find closest market date
                market_df_copy = market_df.copy()
                market_df_copy['date_diff'] = abs(market_df_copy['Date'] - ipo_date)
                closest_idx = market_df_copy['date_diff'].idxmin()
                
                if pd.isna(closest_idx):
                    closest_values.append(np.nan)
                else:
                    closest_values.append(market_df_copy.loc[closest_idx, value_col])
            
            ipo_df[new_col_name] = closest_values
            
            return ipo_df
            
        except Exception as e:
            logger.error(f"Error getting closest market data: {e}")
            return ipo_df
    
    def _merge_sec_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge SEC filing features with IPO data
        
        Args:
            df: IPO DataFrame
            
        Returns:
            DataFrame with merged SEC features
        """
        try:
            if self.sec_features is None or self.sec_features.empty:
                logger.warning("No SEC features available for merging")
                return df
            
            # Ensure CIK columns match
            df['CIK'] = df['CIK'].astype(str).str.zfill(10)
            self.sec_features['cik'] = self.sec_features['cik'].astype(str).str.zfill(10)
            
            # Merge on CIK
            merged = df.merge(
                self.sec_features,
                left_on='CIK',
                right_on='cik',
                how='left',
                suffixes=('', '_sec')
            )
            
            # Check for missing SEC features
            missing_sec = merged['cik'].isna().sum()
            logger.info(f"Companies without SEC features: {missing_sec}")
            
            # Fill missing SEC features with zeros
            sec_feature_cols = [col for col in merged.columns if col.endswith('_sec') or 
                               col.startswith('text_') or col.startswith('financial_') or 
                               col.startswith('sentiment_') or col.startswith('embedding_') or
                               col.startswith('section_')]
            
            for col in sec_feature_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0)
            
            logger.info("SEC features merging completed")
            return merged
            
        except Exception as e:
            logger.error(f"Error merging SEC features: {e}")
            return df
    
    def _final_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final preprocessing steps for combined dataset
        
        Args:
            df: Combined DataFrame
            
        Returns:
            Final preprocessed DataFrame
        """
        try:
            # Remove duplicate columns
            duplicate_cols = df.columns[df.columns.duplicated()]
            if len(duplicate_cols) > 0:
                logger.info(f"Removing {len(duplicate_cols)} duplicate columns")
                df = df.loc[:, ~df.columns.duplicated()]
            
            # Create target variable if possible
            if 'Price' in df.columns and 'close price' in df.columns:
                df['first_day_return'] = (
                    (df['close price'] - df['Price']) / df['Price'] * 100
                )
                logger.info("Target variable 'first_day_return' created")
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # Remove rows with all missing values
            initial_rows = len(df)
            df = df.dropna(how='all')
            if len(df) < initial_rows:
                logger.info(f"Removed {initial_rows - len(df)} rows with all missing values")
            
            logger.info("Final preprocessing completed")
            return df
            
        except Exception as e:
            logger.error(f"Error in final preprocessing: {e}")
            raise
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics for all loaded data
        
        Returns:
            Dictionary with data summary
        """
        summary = {
            'ipo_records': len(self.ipo_data) if self.ipo_data is not None else 0,
            'vix_records': len(self.vix_data) if self.vix_data is not None else 0,
            'fedfunds_records': len(self.fedfunds_data) if self.fedfunds_data is not None else 0,
            'sec_filings': len(self.sec_features) if self.sec_features is not None else 0,
            'combined_records': len(self.combined_data) if self.combined_data is not None else 0
        }
        
        if self.combined_data is not None:
            summary['combined_features'] = len(self.combined_data.columns)
            summary['missing_values'] = self.combined_data.isnull().sum().sum()
        
        return summary
    
    def save_combined_data(self, filepath: str = None):
        """
        Save combined dataset to file
        
        Args:
            filepath: Path to save file (optional)
        """
        if self.combined_data is None:
            logger.warning("No combined data to save")
            return
        
        if filepath is None:
            from config import ENHANCED_DATASET_PATH
            filepath = ENHANCED_DATASET_PATH
        
        try:
            self.combined_data.to_csv(filepath, index=False)
            logger.info(f"Combined data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving combined data: {e}")
            raise
