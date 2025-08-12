"""
SEC Filing Processor Module for IPO Analysis ML Pipeline
=======================================================
Handles processing of SEC filing documents using Gemini API for advanced NLP features
"""

import pandas as pd
import numpy as np
import os
import re
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Gemini API imports
import google.generativeai as genai

# Basic NLP imports (kept for fallback and text preprocessing)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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
    Processor for SEC filing documents using Gemini API for advanced NLP features
    """
    
    def __init__(self):
        """Initialize the SEC filing processor with Gemini API"""
        self._initialize_gemini()
        self.processed_features = {}
        
        logger.info("SEC Filing Processor initialized with Gemini API")
    
    def _initialize_gemini(self):
        """Initialize Gemini API client"""
        try:
            api_key = NLP_CONFIG.get('gemini_api_key')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Initialize model
            self.gemini_model = genai.GenerativeModel(
                model_name=NLP_CONFIG.get('gemini_model', 'gemini-1.5-flash'),
                generation_config=genai.types.GenerationConfig(
                    temperature=NLP_CONFIG.get('gemini_temperature', 0.1),
                    max_output_tokens=NLP_CONFIG.get('gemini_max_tokens', 4096)
                )
            )
            
            logger.info("Gemini API initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {e}")
            raise
    
    def process_all_filings(self, ipo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all SEC filings for IPO companies using Gemini API
        
        Args:
            ipo_df: DataFrame with IPO details
            
        Returns:
            DataFrame with extracted features
        """
        try:
            logger.info(f"Processing filings for {len(ipo_df)} IPO companies using Gemini API...")
            
            processed_data = []
            
            for idx, row in ipo_df.iterrows():
                try:
                    filing_type = str(row['filing'])
                    cik = str(row['CIK'])
                    
                    logger.info(f"Processing {idx+1}/{len(ipo_df)}: {filing_type}/{cik}")
                    
                    # Process filing with Gemini API
                    features = self.process_filing_with_gemini(filing_type, cik)
                    
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
                logger.info(f"Successfully processed {len(features_df)} filings with Gemini API")
                return features_df
            else:
                logger.warning("No filings were successfully processed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error processing filings: {e}")
            raise
    
    def process_filing_with_gemini(self, filing_type: str, cik: str) -> Dict:
        """
        Process a single SEC filing using Gemini API for advanced feature extraction
        
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
            
            # Use Gemini API for advanced analysis
            gemini_features = self.analyze_with_gemini(cleaned_text, sections, filing_type, cik)
            
            # Combine all features
            features = {
                'filing_type': filing_type,
                'cik': cik,
                'gemini_features': gemini_features,
                'sections': {k: len(v) for k, v in sections.items() if v}
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing filing {filing_type}/{cik}: {e}")
            return {}
    
    def analyze_with_gemini(self, full_text: str, sections: Dict[str, str], filing_type: str, cik: str) -> Dict:
        """
        Use Gemini API to analyze SEC filing content with a single comprehensive prompt
        
        Args:
            full_text: Full cleaned filing text
            sections: Extracted sections
            filing_type: Type of filing
            cik: Company CIK
            
        Returns:
            Dictionary with Gemini-analyzed features
        """
        try:
            # Create a comprehensive analysis prompt that covers all aspects
            comprehensive_prompt = self._create_comprehensive_analysis_prompt(filing_type, cik)
            
            # Create a temporary file for upload
            temp_file_path = self._create_temp_filing_file(full_text, filing_type, cik)
            
            try:
                # Make single API call with file upload
                gemini_features = self._call_gemini_with_file(comprehensive_prompt, temp_file_path)
                
                # Parse the comprehensive response
                parsed_features = self._parse_comprehensive_response(gemini_features)
                
                return parsed_features
                
            finally:
                # Clean up temporary file
                self._cleanup_temp_file(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            return self._create_default_features()
    
    def _create_comprehensive_analysis_prompt(self, filing_type: str, cik: str) -> str:
        """Create a single comprehensive prompt for all analysis aspects"""
        return f"""
        You are a senior financial analyst specializing in SEC filing analysis. Analyze the uploaded SEC filing document for company {cik} (filing type: {filing_type}) and provide a comprehensive analysis covering all aspects below.

        Please analyze the document and provide your assessment in the following JSON format. Be thorough but concise in your analysis:

        {{
            "overall_sentiment": {{
                "overall_sentiment": "positive/negative/neutral",
                "confidence_score": 0.0-1.0,
                "risk_awareness": "high/medium/low",
                "transparency_level": "high/medium/low",
                "professional_tone": "excellent/good/fair/poor",
                "credibility_score": 0.0-1.0
            }},
            "risk_assessment": {{
                "risk_severity": "high/medium/low",
                "primary_risk_categories": ["category1", "category2"],
                "risk_mitigation_mentioned": true/false,
                "risk_transparency": "high/medium/low",
                "overall_risk_score": 0.0-1.0
            }},
            "business_analysis": {{
                "business_model_clarity": "high/medium/low",
                "market_opportunity_size": "large/medium/small",
                "competitive_advantages": ["advantage1", "advantage2"],
                "growth_potential": "high/medium/low",
                "business_maturity": "startup/growth/mature"
            }},
            "financial_indicators": {{
                "revenue_growth": "strong/moderate/weak",
                "profitability": "profitable/breakeven/unprofitable",
                "financial_stability": "stable/growing/declining",
                "cash_flow_health": "positive/neutral/negative",
                "financial_risk_level": "low/medium/high"
            }},
            "market_positioning": {{
                "market_position": "leader/challenger/follower",
                "competitive_strength": "strong/moderate/weak",
                "market_penetration": "high/medium/low",
                "industry_positioning": "innovator/established/niche",
                "market_awareness": "high/medium/low"
            }},
            "technology_assessment": {{
                "technology_sophistication": "cutting_edge/advanced/standard/basic",
                "innovation_level": "high/medium/low",
                "ip_protection": "strong/moderate/weak",
                "tech_advantage_sustainability": "long_term/medium_term/short_term",
                "technology_readiness": "mature/developing/research"
            }},
            "compliance_assessment": {{
                "compliance_readiness": "ready/partially_ready/not_ready",
                "regulatory_risk": "low/medium/high",
                "compliance_infrastructure": "robust/adequate/weak",
                "approval_status": "approved/pending/unknown",
                "compliance_score": 0.0-1.0
            }}
        }}

        Focus on:
        1. **Sentiment & Tone**: Overall document sentiment, confidence, risk awareness, transparency, and credibility
        2. **Risk Factors**: Identify and assess risk severity, categories, mitigation strategies, and transparency
        3. **Business Model**: Evaluate business clarity, market opportunity, competitive advantages, and growth potential
        4. **Financial Health**: Assess revenue trends, profitability, stability, cash flow, and financial risk
        5. **Market Position**: Analyze competitive landscape, market penetration, industry positioning, and awareness
        6. **Technology**: Evaluate innovation level, IP protection, technology advantages, and readiness
        7. **Compliance**: Assess regulatory readiness, risk exposure, infrastructure, and approval status

        Provide your analysis based on the content of the uploaded document. Be objective and analytical in your assessment.
        """
    
    def _create_temp_filing_file(self, text: str, filing_type: str, cik: str) -> str:
        """Create a temporary file for Gemini API upload"""
        try:
            # Create temporary file with appropriate extension
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                prefix=f'sec_filing_{filing_type}_{cik}_',
                delete=False,
                encoding='utf-8'
            )
            
            # Write the filing text
            temp_file.write(text)
            temp_file.close()
            
            logger.info(f"Created temporary file: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creating temporary file: {e}")
            raise
    
    def _call_gemini_with_file(self, prompt: str, file_path: str) -> str:
        """
        Make a call to Gemini API with file upload
        
        Args:
            prompt: The analysis prompt
            file_path: Path to the temporary file to upload
            
        Returns:
            Gemini API response
        """
        try:
            # Create file part for upload
            file_part = genai.types.FileData(
                mime_type='text/plain',
                file_uri=file_path
            )
            
            # Generate content with file
            response = self.gemini_model.generate_content([prompt, file_part])
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API call with file failed: {e}")
            return ""
    
    def _parse_comprehensive_response(self, response: str) -> Dict:
        """Parse the comprehensive Gemini response into structured features"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
                
                # Validate and structure the response
                structured_features = {}
                
                # Map each analysis section
                section_mappings = {
                    'overall_sentiment': 'overall_sentiment',
                    'risk_assessment': 'risk_assessment',
                    'business_analysis': 'business_analysis',
                    'financial_indicators': 'financial_indicators',
                    'market_positioning': 'market_positioning',
                    'technology_assessment': 'technology_assessment',
                    'compliance_assessment': 'compliance_assessment'
                }
                
                for response_key, feature_key in section_mappings.items():
                    if response_key in parsed_response:
                        structured_features[feature_key] = parsed_response[response_key]
                    else:
                        # Use default values if section is missing
                        structured_features[feature_key] = self._get_default_section(feature_key)
                
                return structured_features
            else:
                logger.warning("No JSON found in Gemini response, using default features")
                return self._create_default_features()
                
        except Exception as e:
            logger.error(f"Error parsing comprehensive response: {e}")
            return self._create_default_features()
    
    def _get_default_section(self, section_name: str) -> Dict:
        """Get default values for a specific analysis section"""
        default_methods = {
            'overall_sentiment': self._create_default_sentiment,
            'risk_assessment': self._create_default_risk,
            'business_analysis': self._create_default_business,
            'financial_indicators': self._create_default_financial,
            'market_positioning': self._create_default_market,
            'technology_assessment': self._create_default_technology,
            'compliance_assessment': self._create_default_compliance
        }
        
        method = default_methods.get(section_name)
        return method() if method else {}
    
    def _create_default_features(self) -> Dict:
        """Create default features structure when analysis fails"""
        return {
            'overall_sentiment': self._create_default_sentiment(),
            'risk_assessment': self._create_default_risk(),
            'business_analysis': self._create_default_business(),
            'financial_indicators': self._create_default_financial(),
            'market_positioning': self._create_default_market(),
            'technology_assessment': self._create_default_technology(),
            'compliance_assessment': self._create_default_compliance()
        }
    
    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary file after processing"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary file {file_path}: {e}")
    
    def _create_default_sentiment(self) -> Dict:
        """Create default sentiment values"""
        return {
            "overall_sentiment": "neutral",
            "confidence_score": 0.5,
            "risk_awareness": "medium",
            "transparency_level": "medium",
            "professional_tone": "good",
            "credibility_score": 0.5
        }
    
    def _create_default_risk(self) -> Dict:
        """Create default risk values"""
        return {
            "risk_severity": "medium",
            "primary_risk_categories": ["general"],
            "risk_mitigation_mentioned": False,
            "risk_transparency": "medium",
            "overall_risk_score": 0.5
        }
    
    def _create_default_business(self) -> Dict:
        """Create default business values"""
        return {
            "business_model_clarity": "medium",
            "market_opportunity_size": "medium",
            "competitive_advantages": ["standard"],
            "growth_potential": "medium",
            "business_maturity": "growth"
        }
    
    def _create_default_financial(self) -> Dict:
        """Create default financial values"""
        return {
            "revenue_growth": "moderate",
            "profitability": "breakeven",
            "financial_stability": "stable",
            "cash_flow_health": "neutral",
            "financial_risk_level": "medium"
        }
    
    def _create_default_market(self) -> Dict:
        """Create default market values"""
        return {
            "market_position": "follower",
            "competitive_strength": "moderate",
            "market_penetration": "medium",
            "industry_positioning": "established",
            "market_awareness": "medium"
        }
    
    def _create_default_technology(self) -> Dict:
        """Create default technology values"""
        return {
            "technology_sophistication": "standard",
            "innovation_level": "medium",
            "ip_protection": "moderate",
            "tech_advantage_sustainability": "medium_term",
            "technology_readiness": "developing"
        }
    
    def _create_default_compliance(self) -> Dict:
        """Create default compliance values"""
        return {
            "compliance_readiness": "partially_ready",
            "regulatory_risk": "medium",
            "compliance_infrastructure": "adequate",
            "approval_status": "unknown",
            "compliance_score": 0.5
        }
    
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
            
            # Section lengths
            if 'sections' in features:
                for key, value in features['sections'].items():
                    flat_features[f'section_{key}_length'] = value
            
            # Gemini features (flatten)
            if 'gemini_features' in features:
                gemini_features = features['gemini_features']
                
                # Flatten sentiment features
                if 'overall_sentiment' in gemini_features:
                    sentiment = gemini_features['overall_sentiment']
                    for key, value in sentiment.items():
                        flat_features[f'gemini_sentiment_{key}'] = value
                
                # Flatten risk features
                if 'risk_assessment' in gemini_features:
                    risk = gemini_features['risk_assessment']
                    for key, value in risk.items():
                        flat_features[f'gemini_risk_{key}'] = value
                
                # Flatten business features
                if 'business_analysis' in gemini_features:
                    business = gemini_features['business_analysis']
                    for key, value in business.items():
                        flat_features[f'gemini_business_{key}'] = value
                
                # Flatten financial features
                if 'financial_indicators' in gemini_features:
                    financial = gemini_features['financial_indicators']
                    for key, value in financial.items():
                        flat_features[f'gemini_financial_{key}'] = value
                
                # Flatten market features
                if 'market_positioning' in gemini_features:
                    market = gemini_features['market_positioning']
                    for key, value in market.items():
                        flat_features[f'gemini_market_{key}'] = value
                
                # Flatten technology features
                if 'technology_assessment' in gemini_features:
                    tech = gemini_features['technology_assessment']
                    for key, value in tech.items():
                        flat_features[f'gemini_tech_{key}'] = value
                
                # Flatten compliance features
                if 'compliance_assessment' in gemini_features:
                    compliance = gemini_features['compliance_assessment']
                    for key, value in compliance.items():
                        flat_features[f'gemini_compliance_{key}'] = value
            
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
