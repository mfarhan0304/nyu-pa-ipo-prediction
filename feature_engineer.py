"""
Feature Engineering Module for IPO Analysis ML Pipeline
======================================================
Handles feature creation, selection, and preprocessing for machine learning models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

from config import FEATURE_CONFIG, RANDOM_STATE

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for IPO analysis pipeline
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.pca = None
        self.feature_names = []
        self.selected_features = []
        
        logger.info("FeatureEngineer initialized")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the combined dataset
        
        Args:
            df: Combined DataFrame with all data sources
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("Starting feature engineering...")
            
            # Convert lists to strings first (if any)
            df = self._convert_lists_to_strings(df)
            
            # Preprocess comma-separated strings (keyword extraction, TF-IDF, counts)
            df = self._preprocess_comma_separated_strings(df, method='all')
            
            # Create enhanced features
            df = self._create_enhanced_features(df)
            
            # Create interaction features
            df = self._create_interaction_features(df)
            
            # Create derived features
            df = self._create_derived_features(df)
            
            # Handle categorical features
            df = self._encode_categorical_features(df)
            
            # Feature selection
            df = self._select_features(df)
            
            logger.info(f"Feature engineering completed. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def preprocess_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for prediction using fitted transformers
        
        Args:
            df: New DataFrame for prediction
            
        Returns:
            DataFrame preprocessed for prediction
        """
        try:
            logger.info("Preprocessing data for prediction...")
            
            # Make a copy to avoid modifying original data
            df = df.copy()
            
            # Convert lists to strings first (if any)
            df = self._convert_lists_to_strings(df)
            
            # Preprocess comma-separated strings using fitted transformers
            df = self._preprocess_for_prediction_strings(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Apply fitted categorical encoders
            df = self._apply_fitted_encoders(df)
            
            # Select features (ensure same columns as training)
            df = self._select_features_for_prediction(df)
            
            logger.info(f"Prediction preprocessing completed. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in prediction preprocessing: {e}")
            raise
    
    def _preprocess_for_prediction_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess comma-separated strings for prediction using fitted transformers
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with preprocessed features
        """
        try:
            logger.info("Preprocessing comma-separated strings for prediction...")
            
            # Apply keyword extraction (same logic as training)
            df = self._extract_keywords_from_strings(df)
            
            # Apply TF-IDF using fitted vectorizers
            if hasattr(self, 'tfidf_vectorizers'):
                for col, vectorizer in self.tfidf_vectorizers.items():
                    if col in df.columns:
                        try:
                            # Prepare text data
                            text_data = df[col].fillna('').astype(str)
                            
                            # Transform using fitted vectorizer
                            tfidf_features = vectorizer.transform(text_data)
                            
                            # Convert to DataFrame
                            feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_features.shape[1])]
                            tfidf_df = pd.DataFrame(
                                tfidf_features.toarray(),
                                columns=feature_names,
                                index=df.index
                            )
                            
                            # Concatenate with original DataFrame
                            df = pd.concat([df, tfidf_df], axis=1)
                            
                            # Remove original column
                            df = df.drop(columns=[col])
                            
                            logger.info(f"Applied fitted TF-IDF for column {col}")
                            
                        except Exception as e:
                            logger.warning(f"Could not apply TF-IDF for column {col}: {e}")
                            continue
            
            # Apply count features (same logic as training)
            df = self._create_count_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing strings for prediction: {e}")
            return df
    
    def _apply_fitted_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted label encoders to new data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        try:
            logger.info("Applying fitted encoders...")
            
            for col, le in self.label_encoders.items():
                if col in df.columns:
                    try:
                        # Handle missing values
                        df[col] = df[col].fillna('Unknown')
                        
                        # Apply fitted encoder
                        df[f'{col}_encoded'] = le.transform(df[col].astype(str))
                        
                        # Remove original column
                        df = df.drop(columns=[col])
                        
                        logger.info(f"Applied fitted encoder for column {col}")
                        
                    except Exception as e:
                        logger.warning(f"Could not apply encoder for column {col}: {e}")
                        continue
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying fitted encoders: {e}")
            return df
    
    def _select_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features for prediction ensuring same columns as training
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with correct features for prediction
        """
        try:
            if hasattr(self, 'feature_names') and self.feature_names:
                # Select only the features used during training
                available_features = [col for col in self.feature_names if col in df.columns]
                missing_features = [col for col in self.feature_names if col not in df.columns]
                
                if missing_features:
                    logger.warning(f"Missing features for prediction: {missing_features}")
                    # Add missing features with default values
                    for col in missing_features:
                        df[col] = 0
                
                # Select features in the same order as training
                df = df[self.feature_names]
                
                logger.info(f"Selected {len(self.feature_names)} features for prediction")
            else:
                logger.warning("No feature names stored from training. Using all available features.")
            
            return df
            
        except Exception as e:
            logger.error(f"Error selecting features for prediction: {e}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def _create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced features combining traditional and NLP features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with enhanced features
        """
        try:
            logger.info("Creating enhanced features...")
            
            # Text complexity features
            if 'text_word_count' in df.columns and 'text_sentence_count' in df.columns:
                df['text_complexity'] = (
                    df['text_word_count'] / 
                    df['text_sentence_count'].replace(0, 1)
                )
                
                df['text_diversity'] = (
                    df['text_unique_words'] / 
                    df['text_word_count'].replace(0, 1)
                )
            
            # Sentiment-based features
            sentiment_cols = [col for col in df.columns if 'sentiment' in col and 'positive' in col]
            if sentiment_cols:
                df['avg_positive_sentiment'] = df[sentiment_cols].mean(axis=1)
                
                # Risk sentiment (negative sentiment as risk indicator)
                risk_sentiment_cols = [col for col in df.columns if 'sentiment' in col and 'negative' in col]
                if risk_sentiment_cols:
                    df['risk_sentiment_score'] = df[risk_sentiment_cols].mean(axis=1)
            
            # Financial risk indicators
            if 'financial_risk_mentions' in df.columns and 'text_word_count' in df.columns:
                df['risk_intensity'] = (
                    df['financial_risk_mentions'] / 
                    (df['text_word_count'] + 1) * 1000
                )
            
            # Technology and innovation indicators
            if 'financial_technology_mentions' in df.columns and 'text_word_count' in df.columns:
                df['tech_innovation_score'] = (
                    df['financial_technology_mentions'] / 
                    (df['text_word_count'] + 1) * 1000
                )
            
            # Market competition indicators
            if 'financial_market_mentions' in df.columns and 'text_word_count' in df.columns:
                df['market_awareness'] = (
                    df['financial_market_mentions'] / 
                    (df['text_word_count'] + 1) * 1000
                )
            
            # Regulatory compliance indicators
            if 'financial_regulation_mentions' in df.columns and 'text_word_count' in df.columns:
                df['regulatory_compliance'] = (
                    df['financial_regulation_mentions'] / 
                    (df['text_word_count'] + 1) * 1000
                )
            
            # Growth potential indicators
            if 'financial_growth_mentions' in df.columns and 'text_word_count' in df.columns:
                df['growth_potential'] = (
                    df['financial_growth_mentions'] / 
                    (df['text_word_count'] + 1) * 1000
                )
            
            # Document quality indicators
            if 'text_total_length' in df.columns:
                df['document_quality'] = np.where(
                    df['text_total_length'] > df['text_total_length'].median(),
                    1, 0
                )
            
            # Filing type indicators
            if 'filing' in df.columns:
                df['filing_type_encoded'] = pd.Categorical(df['filing']).codes
            
            logger.info("Enhanced features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating enhanced features: {e}")
            raise
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        try:
            logger.info("Creating interaction features...")
            
            # Price and shares interactions
            if 'Price' in df.columns and 'Shares' in df.columns:
                df['price_shares_interaction'] = df['Price'] * df['Shares']
                df['log_price_shares'] = np.log1p(df['price_shares_interaction'])
            
            # Employee and size interactions
            if 'Employees' in df.columns and 'Shares' in df.columns:
                df['employee_size_ratio'] = df['Employees'] / (df['Shares'] + 1)
                df['log_employees'] = np.log1p(df['Employees'])
            
            # Market condition interactions
            if 'VIX' in df.columns and 'FEDFUNDS' in df.columns:
                df['market_volatility_risk'] = df['VIX'] * df['FEDFUNDS']
                df['high_volatility'] = np.where(df['VIX'] > 25, 1, 0)
                df['low_volatility'] = np.where(df['VIX'] < 15, 1, 0)
            
            # Sentiment and risk interactions
            if 'avg_positive_sentiment' in df.columns and 'risk_intensity' in df.columns:
                df['sentiment_risk_balance'] = df['avg_positive_sentiment'] - df['risk_intensity']
            
            logger.info("Interaction features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            raise
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing ones
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        try:
            logger.info("Creating derived features...")
            
            # Time-based features
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['year'] = df['Date'].dt.year
                df['month'] = df['Date'].dt.month
                df['quarter'] = df['Date'].dt.quarter
                df['day_of_week'] = df['Date'].dt.dayofweek
            
            # Financial ratios
            if 'Total Offering Expense' in df.columns and 'Shares' in df.columns:
                df['expense_per_share'] = df['Total Offering Expense'] / (df['Shares'] + 1)
            
            if 'Price' in df.columns and 'Employees' in df.columns:
                df['price_per_employee'] = df['Price'] / (df['Employees'] + 1)
            
            # Market timing features
            if 'year' in df.columns:
                df['post_2008'] = np.where(df['year'] > 2008, 1, 0)
                df['post_2020'] = np.where(df['year'] > 2020, 1, 0)
            
            # Size categories
            # if 'Shares' in df.columns:
            #     df['size_category'] = pd.cut(
            #         df['Shares'], 
            #         bins=[0, 1000000, 5000000, 10000000, np.inf],
            #         labels=['Small', 'Medium', 'Large', 'Very Large']
            #     )
            #     df['size_category_encoded'] = pd.Categorical(df['size_category']).codes
            
            logger.info("Derived features created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating derived features: {e}")
            raise
    
    def _convert_lists_to_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert list values in DataFrame to comma-separated strings
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with list values converted to strings
        """
        try:
            logger.info("Converting list values to strings...")
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if any values are lists
                    has_lists = df[col].apply(lambda x: isinstance(x, list)).any()
                    if has_lists:
                        # Convert lists to comma-separated strings
                        df[col] = df[col].apply(lambda x: ', '.join(str(item) for item in x) if isinstance(x, list) else str(x))
                        logger.info(f"Converted list values in column {col} to comma-separated strings")
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting lists to strings: {e}")
            return df
    
    def _extract_keywords_from_strings(self, df: pd.DataFrame, keyword_columns: List[str] = None) -> pd.DataFrame:
        """
        Extract individual keywords from comma-separated strings and create binary features
        
        Args:
            df: Input DataFrame
            keyword_columns: List of column names to process (if None, auto-detect)
            
        Returns:
            DataFrame with keyword-based binary features
        """
        try:
            logger.info("Extracting keywords from comma-separated strings...")
            
            # Auto-detect keyword columns if not specified
            if keyword_columns is None:
                keyword_columns = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if column contains comma-separated values
                        sample_values = df[col].dropna().head(100)
                        has_commas = sample_values.str.contains(',').any()
                        if has_commas:
                            keyword_columns.append(col)
            
            logger.info(f"Processing keyword columns: {keyword_columns}")
            
            for col in keyword_columns:
                if col not in df.columns:
                    continue
                
                # Get all unique keywords from the column
                all_keywords = set()
                for value in df[col].dropna():
                    if isinstance(value, str) and ',' in value:
                        keywords = [kw.strip().lower() for kw in value.split(',')]
                        all_keywords.update(keywords)
                
                # Create binary features for each keyword
                for keyword in sorted(all_keywords):
                    feature_name = f"{col}_{keyword.replace(' ', '_')}_present"
                    df[feature_name] = df[col].apply(
                        lambda x: 1 if isinstance(x, str) and keyword.lower() in x.lower() else 0
                    )
                
                # Remove original column to avoid duplication
                df = df.drop(columns=[col])
                
                logger.info(f"Created {len(all_keywords)} binary features for column {col}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return df
    
    def _create_tfidf_features(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """
        Create TF-IDF features from comma-separated strings
        
        Args:
            df: Input DataFrame
            text_columns: List of column names to process (if None, auto-detect)
            
        Returns:
            DataFrame with TF-IDF features
        """
        try:
            logger.info("Creating TF-IDF features from comma-separated strings...")
            
            # Auto-detect text columns if not specified
            if text_columns is None:
                text_columns = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if column contains comma-separated values
                        sample_values = df[col].dropna().head(100)
                        has_commas = sample_values.str.contains(',').any()
                        if has_commas:
                            text_columns.append(col)
            
            logger.info(f"Processing TF-IDF columns: {text_columns}")
            
            for col in text_columns:
                if col not in df.columns:
                    continue
                
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    
                    # Prepare text data
                    text_data = df[col].fillna('').astype(str)
                    
                    # Create TF-IDF vectorizer
                    tfidf = TfidfVectorizer(
                        max_features=20,  # Limit features to prevent explosion
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95
                    )
                    
                    # Fit and transform
                    tfidf_features = tfidf.fit_transform(text_data)
                    
                    # Convert to DataFrame
                    feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_features.shape[1])]
                    tfidf_df = pd.DataFrame(
                        tfidf_features.toarray(),
                        columns=feature_names,
                        index=df.index
                    )
                    
                    # Concatenate with original DataFrame
                    df = pd.concat([df, tfidf_df], axis=1)
                    
                    # Store vectorizer for later use
                    if not hasattr(self, 'tfidf_vectorizers'):
                        self.tfidf_vectorizers = {}
                    self.tfidf_vectorizers[col] = tfidf
                    
                    # Remove original column
                    df = df.drop(columns=[col])
                    
                    logger.info(f"Created {len(feature_names)} TF-IDF features for column {col}")
                    
                except Exception as e:
                    logger.warning(f"Could not create TF-IDF features for column {col}: {e}")
                    continue
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF features: {e}")
            return df
    
    def _create_count_features(self, df: pd.DataFrame, count_columns: List[str] = None) -> pd.DataFrame:
        """
        Create count-based features from comma-separated strings
        
        Args:
            df: Input DataFrame
            count_columns: List of column names to process (if None, auto-detect)
            
        Returns:
            DataFrame with count-based features
        """
        try:
            logger.info("Creating count-based features from comma-separated strings...")
            
            # Auto-detect count columns if not specified
            if count_columns is None:
                count_columns = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if column contains comma-separated values
                        sample_values = df[col].dropna().head(100)
                        has_commas = sample_values.str.contains(',').any()
                        if has_commas:
                            count_columns.append(col)
            
            logger.info(f"Processing count columns: {count_columns}")
            
            for col in count_columns:
                if col not in df.columns:
                    continue
                
                # Count of items
                df[f"{col}_item_count"] = df[col].apply(
                    lambda x: len(x.split(',')) if isinstance(x, str) and ',' in x else 1
                )
                
                # Length of string
                df[f"{col}_string_length"] = df[col].apply(
                    lambda x: len(str(x)) if pd.notna(x) else 0
                )
                
                # Has multiple items (binary)
                df[f"{col}_has_multiple"] = df[col].apply(
                    lambda x: 1 if isinstance(x, str) and ',' in x else 0
                )
                
                logger.info(f"Created count features for column {col}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating count features: {e}")
            return df
    
    def _preprocess_comma_separated_strings(self, df: pd.DataFrame, method: str = 'keyword') -> pd.DataFrame:
        """
        Preprocess comma-separated strings using specified method
        
        Args:
            df: Input DataFrame
            method: 'keyword', 'tfidf', 'count', or 'all'
            
        Returns:
            DataFrame with preprocessed features
        """
        try:
            logger.info(f"Preprocessing comma-separated strings using method: {method}")
            
            if method == 'keyword' or method == 'all':
                df = self._extract_keywords_from_strings(df)
            
            if method == 'tfidf' or method == 'all':
                df = self._create_tfidf_features(df)
            
            if method == 'count' or method == 'all':
                df = self._create_count_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing comma-separated strings: {e}")
            return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        try:
            logger.info("Encoding categorical features...")
            
            # First convert any list values to strings
            df = self._convert_lists_to_strings(df)
            
            # Identify categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                if col in df.columns and df[col].nunique() < 100:  # Only encode if reasonable number of categories
                    try:
                        # Handle missing values
                        df[col] = df[col].fillna('Unknown')
                        
                        # Create label encoder
                        le = LabelEncoder()
                        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                        
                        # Store encoder for later use
                        self.label_encoders[col] = le
                        
                        logger.info(f"Encoded categorical column: {col}")
                        
                    except Exception as e:
                        logger.warning(f"Could not encode column {col}: {e}")
                        # Try to handle the column by converting everything to strings first
                        try:
                            logger.info(f"Attempting to fix column {col} by converting all values to strings...")
                            df[col] = df[col].astype(str)
                            le = LabelEncoder()
                            df[f'{col}_encoded'] = le.fit_transform(df[col])
                            self.label_encoders[col] = le
                            logger.info(f"Successfully encoded column {col} after conversion to strings")
                        except Exception as e2:
                            logger.warning(f"Could not encode column {col} even after string conversion: {e2}")
                            continue
            
            logger.info("Categorical feature encoding completed")
            return df
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {e}")
            raise
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features for modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with selected features
        """
        try:
            logger.info("Selecting features for modeling...")
            
            # Get all feature categories
            all_features = []
            
            # Traditional features
            traditional_features = [col for col in FEATURE_CONFIG['traditional_features'] if col in df.columns]
            all_features.extend(traditional_features)
            
            # NLP features
            nlp_features = [col for col in FEATURE_CONFIG['nlp_features'] if col in df.columns]
            all_features.extend(nlp_features)
            
            # Financial features
            financial_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_CONFIG['financial_features_prefixes'])]
            all_features.extend(financial_features)
            
            # Section features
            section_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_CONFIG['section_features_prefixes'])]
            all_features.extend(section_features)
            
            # Embedding features
            embedding_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in FEATURE_CONFIG['embedding_features_prefixes'])]
            all_features.extend(embedding_features)
            
            # Additional engineered features
            engineered_features = [col for col in df.columns if col not in all_features and col not in ['Date', 'Symbol', 'Company Name', 'CIK', 'cik']]
            all_features.extend(engineered_features)
            
            # Remove duplicates
            all_features = list(set(all_features))
            
            # Filter to available columns
            available_features = [col for col in all_features if col in df.columns]
            
            logger.info(f"Selected {len(available_features)} features for modeling")
            
            # Store feature names
            self.feature_names = available_features
            self.selected_features = available_features
            
            # Select only the chosen features
            selected_df = df[available_features].copy()
            
            # Handle missing values
            selected_df = selected_df.fillna(0)
            
            # Convert to numeric
            for col in selected_df.columns:
                selected_df[col] = pd.to_numeric(selected_df[col], errors='coerce').fillna(0)
            
            logger.info("Feature selection completed successfully")
            return selected_df
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled training and test features
        """
        try:
            logger.info("Scaling features...")
            
            # Fit scaler on training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Transform test data if provided
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                logger.info("Feature scaling completed successfully")
                return X_train_scaled, X_test_scaled
            else:
                logger.info("Feature scaling completed successfully")
                return X_train_scaled, None
                
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise
    
    def apply_feature_selection(self, X: np.ndarray, y: np.ndarray, k: int = 150) -> np.ndarray:
        """
        Apply feature selection using statistical tests
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            Selected feature matrix
        """
        try:
            logger.info(f"Applying feature selection (k={k})...")
            
            # Create feature selector
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            
            # Fit and transform
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support()
            selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_indices[i]]
            
            logger.info(f"Feature selection completed. Selected {X_selected.shape[1]} features")
            logger.info(f"Selected features: {selected_feature_names[:10]}...")  # Show first 10
            
            return X_selected
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise
    
    def fit_pca(self, X: np.ndarray, n_components: float = 0.95) -> np.ndarray:
        """
        Fit PCA on training data for dimensionality reduction
        
        Args:
            X: Training feature matrix
            n_components: Number of components or explained variance ratio
            
        Returns:
            PCA-transformed training feature matrix
        """
        try:
            logger.info(f"Fitting PCA on training data (n_components={n_components})...")
            
            # Create and fit PCA on training data
            self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
            X_pca = self.pca.fit_transform(X)
            
            logger.info(f"PCA fitted on training data. Reduced from {X.shape[1]} to {X_pca.shape[1]} features")
            logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
            
            return X_pca
            
        except Exception as e:
            logger.error(f"Error in fitting PCA: {e}")
            raise
    
    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA (for test data)
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            PCA-transformed feature matrix
        """
        try:
            if self.pca is None:
                raise ValueError("PCA must be fitted first using fit_pca() before calling transform_pca()")
            
            logger.info(f"Transforming data using fitted PCA...")
            
            # Transform using fitted PCA
            X_pca = self.pca.transform(X)
            
            logger.info(f"PCA transformation completed. Transformed from {X.shape[1]} to {X_pca.shape[1]} features")
            
            return X_pca
            
        except Exception as e:
            logger.error(f"Error in PCA transformation: {e}")
            raise
    
    def get_feature_importance_scores(self) -> pd.DataFrame:
        """
        Get feature importance scores from feature selection
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_selector is None:
            logger.warning("No feature selector available")
            return pd.DataFrame()
        
        try:
            # Get scores and p-values
            scores = self.feature_selector.scores_
            p_values = self.feature_selector.pvalues_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'score': scores,
                'p_value': p_values
            })
            
            # Sort by score
            importance_df = importance_df.sort_values('score', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance scores: {e}")
            return pd.DataFrame()
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of feature engineering process
        
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_features': len(self.feature_names),
            'selected_features': len(self.selected_features),
            'categorical_encoders': len(self.label_encoders),
            'scaler_fitted': hasattr(self.scaler, 'mean_'),
            'feature_selector_fitted': self.feature_selector is not None,
            'pca_fitted': self.pca is not None
        }
        
        if self.pca is not None:
            summary['pca_components'] = self.pca.n_components_
            summary['explained_variance'] = self.pca.explained_variance_ratio_.sum()
            summary['pca_fitted'] = hasattr(self.pca, 'components_')
        
        return summary
