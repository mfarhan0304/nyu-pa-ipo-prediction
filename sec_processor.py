"""
SEC Filing Processor Module for IPO Analysis ML Pipeline
=======================================================
Handles processing of SEC filing documents and extraction of NLP features
"""

import pandas as pd
import numpy as np
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# NLP imports
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config import NLP_CONFIG, SEC_FILINGS_DIR

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("NLTK data download failed - some features may not work")

class SECFilingProcessor:
    """
    Processor for SEC filing documents to extract NLP features
    """
    
    def __init__(self):
        """Initialize the SEC filing processor"""
        self._initialize_models()
        self.processed_features = {}
        
        logger.info("SEC Filing Processor initialized")
    
    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            # TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=NLP_CONFIG['max_features'],
                stop_words='english',
                ngram_range=(1, 2),
                # min_df=2,
                # max_df=0.95
            )
            
            # LDA topic model
            self.lda_model = LatentDirichletAllocation(
                n_components=5,
                random_state=42,
                max_iter=50
            )
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            raise
    
    def process_all_filings(self, ipo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all SEC filings for IPO companies
        
        Args:
            ipo_df: DataFrame with IPO details
            
        Returns:
            DataFrame with extracted features
        """
        try:
            logger.info(f"Processing filings for {len(ipo_df)} IPO companies...")
            
            processed_data = []
            
            processed_count = 0  # Local counter for accurate progress tracking
            
            for idx, row in ipo_df.iterrows():
                try:
                    processed_count += 1  # Increment local counter
                    
                    # Safety check: ensure we don't exceed the intended limit
                    if processed_count > len(ipo_df):
                        logger.warning(f"Reached processing limit of {len(ipo_df)}. Stopping.")
                        break
                    
                    filing_type = str(row['filing'])
                    cik = str(row['CIK'])
                    
                    logger.info(f"Processing {processed_count}/{len(ipo_df)}: {filing_type}/{cik}")
                    
                    # Process filing
                    features = self.process_filing(filing_type, cik)
                    
                    if features:
                        # Flatten features for DataFrame
                        flat_features = self._flatten_features(features)
                        flat_features['ipo_index'] = idx
                        flat_features['symbol'] = row.get('Symbol', '')
                        flat_features['company_name'] = row.get('Company Name', '')
                        
                        processed_data.append(flat_features)
                    
                except Exception as e:
                    logger.error(f"Error processing filing {row.get('filing', '')}/{row.get('CIK', '')}: {e}")
                    continue
            
            # Create DataFrame
            if processed_data:
                features_df = pd.DataFrame(processed_data)
                actual_processed = len(features_df)
                expected_processed = len(ipo_df)
                
                logger.info(f"Successfully processed {actual_processed}/{expected_processed} filings")
                
                # Validate that we didn't exceed the limit
                if actual_processed > expected_processed:
                    logger.warning(f"Processed more filings than expected: {actual_processed} > {expected_processed}")
                elif actual_processed < expected_processed:
                    logger.info(f"Some filings failed to process: {actual_processed}/{expected_processed}")
                
                return features_df
            else:
                logger.warning("No filings were successfully processed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error processing filings: {e}")
            raise
    
    def process_filing(self, filing_type: str, cik: str) -> Dict:
        """
        Process a single SEC filing and extract features
        
        Args:
            filing_type: Type of filing (F1, S1, etc.)
            cik: Company CIK number
            
        Returns:
            Dictionary with extracted features
        """
        try:
            # Read filing content
            content = self.read_sec_filing(filing_type, cik)
            if not content:
                return {}
            
            # Preprocess text
            cleaned_text = self.preprocess_text(content)
            
            # Extract sections
            sections = self.extract_sections(cleaned_text)
            
            # Generate embeddings for full text and key sections
            embeddings = {}
            for section_name, section_text in sections.items():
                if section_text:
                    embeddings[f'{section_name}_embedding'] = self.generate_embeddings(section_text)
            
            # Full text embedding
            embeddings['full_text_embedding'] = self.generate_embeddings(cleaned_text)
            
            # Sentiment analysis for key sections
            sentiment_scores = {}
            for section_name, section_text in sections.items():
                if section_text:
                    sentiment_scores[f'{section_name}_sentiment'] = self.analyze_sentiment(section_text)
            
            # Financial metrics
            financial_metrics = self.extract_financial_metrics(cleaned_text)
            
            # Text statistics
            text_stats = self.calculate_text_statistics(cleaned_text)
            
            # Combine all features
            features = {
                'filing_type': filing_type,
                'cik': cik,
                'embeddings': embeddings,
                'sentiment_scores': sentiment_scores,
                'financial_metrics': financial_metrics,
                'text_stats': text_stats,
                'sections': {k: len(v) for k, v in sections.items() if v}
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing filing {filing_type}/{cik}: {e}")
            return {}
    
    def read_sec_filing(self, filing_type: str, cik: str) -> Optional[str]:
        """
        Read SEC filing text from file
        
        Args:
            filing_type: Type of filing (F1, S1, etc.)
            cik: Company CIK number
            
        Returns:
            Filing text content or None if not found
        """
        try:
            # Construct file path
            filing_dir = SEC_FILINGS_DIR / filing_type
            file_path = filing_dir / f"{cik}.txt"
            
            if not file_path.exists():
                logger.warning(f"Filing not found: {file_path}")
                return None
            
            # Read file content (limit size for performance)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(NLP_CONFIG['max_text_length'])
            
            logger.info(f"Successfully read filing: {file_path} ({len(content):,} characters)")
            return content
            
        except Exception as e:
            logger.error(f"Error reading filing {filing_type}/{cik}: {e}")
            return None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess and clean filing text
        
        Args:
            text: Raw filing text
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""
        
        try:
            # Remove HTML tags and special characters
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'&[a-zA-Z]+;', '', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove page headers/footers
            text = re.sub(r'Page \d+ of \d+', '', text)
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
            
            # Remove SEC filing metadata
            text = re.sub(r'SECURITIES AND EXCHANGE COMMISSION.*?Washington, D\.C\.\s*\d+', '', text, flags=re.DOTALL)
            
            # Clean up
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text.strip() if text else ""
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key sections from SEC filing
        
        Args:
            text: Preprocessed filing text
            
        Returns:
            Dictionary of section names and content
        """
        sections = {}
        
        try:
            # Use patterns from config
            section_patterns = NLP_CONFIG['section_patterns']
            
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[section_name] = match.group(0).strip()
                else:
                    sections[section_name] = ""
            
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            return {}
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate basic embeddings using TF-IDF
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text or len(text.strip()) < 10:
            return np.zeros(NLP_CONFIG['max_features'])
        
        try:
            # Use TF-IDF for basic embeddings
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            tfidf_array = tfidf_matrix.toarray().flatten()
            
            # Ensure consistent dimensions
            if len(tfidf_array) < NLP_CONFIG['max_features']:
                tfidf_array = np.pad(tfidf_array, (0, NLP_CONFIG['max_features'] - len(tfidf_array)))
            else:
                tfidf_array = tfidf_array[:NLP_CONFIG['max_features']]
            
            return tfidf_array
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.zeros(NLP_CONFIG['max_features'])
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or len(text.strip()) < 10:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        try:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text[:NLP_CONFIG['sentiment_text_limit']])
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert polarity to sentiment distribution
            if polarity > 0.1:
                return {'positive': 0.7, 'negative': 0.1, 'neutral': 0.2}
            elif polarity < -0.1:
                return {'positive': 0.1, 'negative': 0.7, 'neutral': 0.2}
            else:
                return {'positive': 0.2, 'negative': 0.2, 'neutral': 0.6}
                    
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
    
    def extract_financial_metrics(self, text: str) -> Dict[str, float]:
        """
        Extract financial metrics and indicators from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of financial metrics
        """
        metrics = {
            'revenue_mentions': 0,
            'debt_mentions': 0,
            'growth_mentions': 0,
            'risk_mentions': 0,
            'technology_mentions': 0,
            'regulation_mentions': 0,
            'market_mentions': 0
        }
        
        if not text:
            return metrics
        
        try:
            text_lower = text.lower()
            financial_keywords = NLP_CONFIG['financial_keywords']
            
            # Count keyword mentions
            for category, keywords in financial_keywords.items():
                count = sum(text_lower.count(keyword) for keyword in keywords)
                metrics[f'{category}_mentions'] = count
            
            # Normalize by text length
            text_length = len(text.split())
            if text_length > 0:
                for key in metrics:
                    metrics[key] = metrics[key] / text_length * 1000  # Per 1000 words
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
            return metrics
    
    def calculate_text_statistics(self, text: str) -> Dict[str, float]:
        """
        Calculate text statistics
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        try:
            sentences = sent_tokenize(text)
            words = text.split()
            
            stats = {
                'total_length': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_sentence_length': len(words) / max(len(sentences), 1),
                'unique_words': len(set(word.lower() for word in words))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating text statistics: {e}")
            return {
                'total_length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'unique_words': 0
            }
    
    def _flatten_features(self, features: Dict) -> Dict:
        """
        Flatten nested feature dictionary for DataFrame creation
        
        Args:
            features: Nested features dictionary
            
        Returns:
            Flattened dictionary
        """
        flat_features = {}
        
        try:
            # Basic info
            flat_features.update({
                'filing_type': features.get('filing_type', ''),
                'cik': features.get('cik', '')
            })
            
            # Text statistics
            if 'text_stats' in features:
                for key, value in features['text_stats'].items():
                    flat_features[f'text_{key}'] = value
            
            # Financial metrics
            if 'financial_metrics' in features:
                for key, value in features['financial_metrics'].items():
                    flat_features[f'financial_{key}'] = value
            
            # Section lengths
            if 'sections' in features:
                for key, value in features['sections'].items():
                    flat_features[f'section_{key}_length'] = value
            
            # Sentiment scores (flatten)
            if 'sentiment_scores' in features:
                for section_name, sentiment in features['sentiment_scores'].items():
                    for sentiment_type, score in sentiment.items():
                        flat_features[f'sentiment_{section_name}_{sentiment_type}'] = score
            
            # Embeddings (take first few dimensions for DataFrame)
            if 'embeddings' in features:
                for embedding_name, embedding_vector in features['embeddings'].items():
                    if isinstance(embedding_vector, np.ndarray) and len(embedding_vector) > 0:
                        # Take first 20 dimensions for DataFrame
                        for i in range(min(20, len(embedding_vector))):
                            flat_features[f'embedding_{embedding_name}_dim_{i}'] = embedding_vector[i]
            
            return flat_features
            
        except Exception as e:
            logger.error(f"Error flattening features: {e}")
            return {}
    
    def save_features(self, features_df: pd.DataFrame, filename: str = "sec_filing_features.csv"):
        """
        Save extracted features to CSV
        
        Args:
            features_df: DataFrame with features
            filename: Output filename
        """
        try:
            features_df.to_csv(filename, index=False)
            logger.info(f"Features saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise
