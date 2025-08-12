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
from pathlib import Path

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
            
            # Define target variables to exclude
            target_variables = ['close_price_target', 'price_direction', 'close price', 'first_day_return']
            
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
            engineered_features = [col for col in df.columns if col not in all_features and col not in ['Date', 'Symbol', 'Company Name', 'CIK', 'cik'] + target_variables]
            all_features.extend(engineered_features)
            
            # Remove duplicates
            all_features = list(set(all_features))
            
            # Filter to available columns and exclude target variables
            available_features = [col for col in all_features if col in df.columns and col not in target_variables]
            
            logger.info(f"Selected {len(available_features)} features for modeling")
            logger.info(f"Excluded target variables: {target_variables}")
            
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
    
    def find_optimal_k(self, X: np.ndarray, y: np.ndarray, max_k: int = None, 
                       min_k: int = 100, step: int = 10, plot: bool = True) -> int:
        """
        Find optimal k using scree plot analysis
        
        Args:
            X: Feature matrix
            y: Target variable
            max_k: Maximum k to test (default: min(200, n_features))
            min_k: Minimum k to test
            step: Step size for k values
            plot: Whether to create scree plot
            
        Returns:
            Optimal k value
        """
        try:
            logger.info("Finding optimal k using scree plot analysis...")
            
            # Set max_k if not provided
            if max_k is None:
                max_k = min(200, X.shape[1])
            
            # Generate k values to test
            k_values = list(range(min_k, max_k + 1, step))
            if max_k not in k_values:
                k_values.append(max_k)
            
            # Calculate scores for each k
            scores = []
            feature_importance_data = []
            
            for k in k_values:
                try:
                    # Create selector for this k
                    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
                    selector.fit(X, y)
                    
                    # Get scores and sort them
                    feature_scores = selector.scores_
                    sorted_indices = np.argsort(feature_scores)[::-1]
                    
                    # Calculate cumulative score (sum of top k feature scores)
                    top_k_scores = feature_scores[sorted_indices[:k]]
                    cumulative_score = np.sum(top_k_scores)
                    
                    scores.append(cumulative_score)
                    feature_importance_data.append({
                        'k': k,
                        'cumulative_score': cumulative_score,
                        'top_features': sorted_indices[:k],
                        'top_scores': top_k_scores
                    })
                    
                    logger.info(f"k={k:3d}: Cumulative score = {cumulative_score:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Error calculating score for k={k}: {e}")
                    continue
            
            if not scores:
                logger.warning("Could not calculate scores for any k value")
                return min_k
            
            # Find elbow point using second derivative method
            optimal_k = self._find_elbow_point(k_values, scores)
            
            logger.info(f"Optimal k determined: {optimal_k}")
            
            # Create scree plot if requested
            if plot:
                self._create_scree_plot(k_values, scores, optimal_k, feature_importance_data)
            
            return optimal_k
            
        except Exception as e:
            logger.error(f"Error finding optimal k: {e}")
            return min_k
    
    def _find_elbow_point(self, k_values: List[int], scores: List[float]) -> int:
        """
        Find elbow point using second derivative method
        
        Args:
            k_values: List of k values tested
            scores: List of corresponding scores
            
        Returns:
            k value at the elbow point
        """
        try:
            if len(scores) < 3:
                return k_values[0]
            
            # Calculate first and second derivatives
            first_derivative = np.gradient(scores)
            second_derivative = np.gradient(first_derivative)
            
            # Find point with maximum second derivative (elbow)
            elbow_idx = np.argmax(second_derivative)
            
            # Ensure we don't pick the first or last point
            if elbow_idx == 0:
                elbow_idx = 1
            elif elbow_idx == len(k_values) - 1:
                elbow_idx = len(k_values) - 2
            
            optimal_k = k_values[elbow_idx]
            
            logger.info(f"Elbow point found at k={optimal_k} (second derivative method)")
            return optimal_k
            
        except Exception as e:
            logger.warning(f"Error in elbow point detection: {e}")
            # Fallback: return k with 80% of max score
            max_score = max(scores)
            threshold = 0.8 * max_score
            
            for i, score in enumerate(scores):
                if score >= threshold:
                    return k_values[i]
            
            return k_values[len(k_values) // 2]  # Return middle value as fallback
    
    def _create_scree_plot(self, k_values: List[int], scores: List[float], 
                          optimal_k: int, feature_importance_data: List[Dict]):
        """
        Create scree plot to visualize feature selection results
        
        Args:
            k_values: List of k values tested
            scores: List of corresponding scores
            optimal_k: Optimal k value found
            feature_importance_data: Detailed data for each k
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Cumulative score vs k
            ax1.plot(k_values, scores, 'bo-', linewidth=2, markersize=6)
            ax1.axvline(x=optimal_k, color='red', linestyle='--', 
                       label=f'Optimal k = {optimal_k}')
            ax1.set_xlabel('Number of Features (k)')
            ax1.set_ylabel('Cumulative Feature Importance Score')
            ax1.set_title('Scree Plot: Feature Selection')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Marginal improvement (first derivative)
            if len(scores) > 1:
                marginal_improvement = np.diff(scores)
                ax2.plot(k_values[1:], marginal_improvement, 'go-', linewidth=2, markersize=6)
                ax2.axvline(x=optimal_k, color='red', linestyle='--', 
                           label=f'Optimal k = {optimal_k}')
                ax2.set_xlabel('Number of Features (k)')
                ax2.set_ylabel('Marginal Improvement')
                ax2.set_title('Marginal Improvement vs k')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path("results/feature_selection_scree_plot.png")
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Scree plot saved to {plot_path}")
            
            # Save detailed results
            results_path = Path("results/feature_selection_analysis.csv")
            results_df = pd.DataFrame(feature_importance_data)
            results_df.to_csv(results_path, index=False)
            logger.info(f"Feature selection analysis saved to {results_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot creation")
        except Exception as e:
            logger.warning(f"Error creating scree plot: {e}")
    
    def apply_feature_selection(self, X: np.ndarray, y: np.ndarray, k: int = None, 
                               auto_k: bool = True, max_k: int = 200) -> np.ndarray:
        """
        Apply feature selection using statistical tests with optimal k selection
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select (if auto_k=False)
            auto_k: Whether to automatically find optimal k
            max_k: Maximum k to consider when auto_k=True
            
        Returns:
            Selected feature matrix
        """
        try:
            # Determine optimal k if requested
            if auto_k:
                optimal_k = self.find_optimal_k(X, y, max_k=max_k)
                k = optimal_k
                logger.info(f"Auto-selected optimal k: {k}")
            elif k is None:
                k = min(150, X.shape[1])  # Default fallback
                logger.info(f"Using default k: {k}")
            
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
    
    def get_optimal_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                      max_k: int = 200) -> pd.DataFrame:
        """
        Get feature importance scores using optimal k selection
        
        Args:
            X: Feature matrix
            y: Target variable
            max_k: Maximum k to consider
            
        Returns:
            DataFrame with feature importance scores and selection status
        """
        try:
            logger.info("Getting feature importance with optimal k selection...")
            
            # Find optimal k
            optimal_k = self.find_optimal_k(X, y, max_k=max_k, plot=False)
            
            # Apply feature selection with optimal k
            X_selected = self.apply_feature_selection(X, y, k=optimal_k, auto_k=False)
            
            # Get feature importance scores
            importance_df = self.get_feature_importance_scores()
            
            if not importance_df.empty:
                # Add selection status
                importance_df['selected'] = importance_df.index < optimal_k
                importance_df['rank'] = range(1, len(importance_df) + 1)
                
                # Add optimal k information
                importance_df.attrs['optimal_k'] = optimal_k
                importance_df.attrs['total_features'] = X.shape[1]
                importance_df.attrs['selected_features'] = optimal_k
                
                logger.info(f"Feature importance analysis completed. Optimal k: {optimal_k}")
                logger.info(f"Top 5 features: {importance_df.head()['feature'].tolist()}")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting optimal feature importance: {e}")
            return pd.DataFrame()
    
    def get_feature_selection_summary(self) -> Dict:
        """
        Get summary of feature selection process
        
        Returns:
            Dictionary with feature selection summary
        """
        summary = {
            'feature_selector_fitted': self.feature_selector is not None,
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'selected_features': len(self.selected_features) if self.selected_features else 0
        }
        
        if self.feature_selector is not None:
            summary['selection_method'] = 'SelectKBest with f_regression'
            summary['k_used'] = self.feature_selector.k
            
            # Add feature importance info if available
            if hasattr(self.feature_selector, 'scores_'):
                summary['max_score'] = float(np.max(self.feature_selector.scores_))
                summary['min_score'] = float(np.min(self.feature_selector.scores_))
                summary['mean_score'] = float(np.mean(self.feature_selector.scores_))
        
        return summary
    
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
