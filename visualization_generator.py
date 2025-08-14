"""
Visualization Generator Module for IPO Analysis ML Pipeline
=========================================================
Generates comprehensive visualizations for IPO analysis reports including charts, plots, and interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

from config import RESULTS_DIR, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Generates comprehensive visualizations for IPO analysis reports
    """
    
    def __init__(self):
        """Initialize the visualization generator"""
        self.output_dir = VISUALIZATIONS_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Set figure size defaults
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        logger.info("Visualization Generator initialized")
    
    def generate_comprehensive_report(self, data: pd.DataFrame, model_results: Dict, 
                                   feature_importance: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive visualization report
        
        Args:
            data: Combined IPO dataset
            model_results: Model training results
            feature_importance: Feature importance DataFrame
            
        Returns:
            Dictionary with generated visualization file paths
        """
        try:
            logger.info("Generating comprehensive visualization report...")
            
            generated_files = {}
            
            # 1. Data Overview Visualizations
            logger.info("Generating data overview visualizations...")
            overview_files = self._generate_data_overview(data)
            generated_files.update(overview_files)
            
            # 2. Market Analysis Visualizations
            logger.info("Generating market analysis visualizations...")
            market_files = self._generate_market_analysis(data)
            generated_files.update(market_files)
            
            # 3. Model Performance Visualizations
            logger.info("Generating model performance visualizations...")
            model_files = self._generate_model_performance(model_results)
            generated_files.update(model_files)
            
            # 4. Feature Analysis Visualizations
            if feature_importance is not None:
                logger.info("Generating feature analysis visualizations...")
                feature_files = self._generate_feature_analysis(feature_importance, data)
                generated_files.update(feature_files)
            
            # 5. Interactive Dashboard
            logger.info("Generating interactive dashboard...")
            dashboard_file = self._generate_interactive_dashboard(data, model_results, feature_importance)
            generated_files['interactive_dashboard'] = dashboard_file
            
            # 6. Summary Report
            logger.info("Generating summary report...")
            summary_file = self._generate_summary_report(data, model_results, generated_files)
            generated_files['summary_report'] = summary_file
            
            logger.info(f"Generated {len(generated_files)} visualization files")
            return generated_files
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def _generate_data_overview(self, data: pd.DataFrame) -> Dict:
        """Generate data overview visualizations"""
        files = {}
        
        try:
            # 1. IPO Timeline Distribution
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('IPO Data Overview', fontsize=16, fontweight='bold')
            
            # IPO count by year
            if 'Date' in data.columns:
                data['Year'] = pd.to_datetime(data['Date']).dt.year
                year_counts = data['Year'].value_counts().sort_index()
                axes[0, 0].bar(year_counts.index, year_counts.values, color=self.colors['primary'])
                axes[0, 0].set_title('IPO Count by Year')
                axes[0, 0].set_xlabel('Year')
                axes[0, 0].set_ylabel('Number of IPOs')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Price distribution
            if 'Price' in data.columns:
                axes[0, 1].hist(data['Price'].dropna(), bins=30, color=self.colors['secondary'], alpha=0.7)
                axes[0, 1].set_title('IPO Price Distribution')
                axes[0, 1].set_xlabel('IPO Price ($)')
                axes[0, 1].set_ylabel('Frequency')
            
            # Shares distribution
            if 'Shares' in data.columns:
                axes[1, 0].hist(data['Shares'].dropna(), bins=30, color=self.colors['success'], alpha=0.7)
                axes[1, 0].set_title('Shares Offered Distribution')
                axes[1, 0].set_xlabel('Shares (Millions)')
                axes[1, 0].set_ylabel('Frequency')
            
            # First day return distribution
            if 'first_day_return' in data.columns:
                axes[1, 1].hist(data['first_day_return'].dropna(), bins=30, color=self.colors['info'], alpha=0.7)
                axes[1, 1].set_title('First Day Return Distribution')
                axes[1, 1].set_xlabel('First Day Return (%)')
                axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            filepath = self.output_dir / "ipo_data_overview.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            files['data_overview'] = str(filepath)
            
            # 2. Correlation Heatmap
            if len(data.select_dtypes(include=[np.number]).columns) > 1:
                numeric_data = data.select_dtypes(include=[np.number]).dropna()
                if len(numeric_data) > 0:
                    plt.figure(figsize=(14, 10))
                    correlation_matrix = numeric_data.corr()
                    
                    # Create mask for upper triangle
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                    
                    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    filepath = self.output_dir / "correlation_heatmap.png"
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    files['correlation_heatmap'] = str(filepath)
            
            # 3. Missing Data Visualization
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                plt.figure(figsize=(12, 8))
                missing_data[missing_data > 0].plot(kind='bar', color=self.colors['warning'])
                plt.title('Missing Data by Feature', fontsize=16, fontweight='bold')
                plt.xlabel('Features')
                plt.ylabel('Missing Values Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                filepath = self.output_dir / "missing_data_analysis.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                files['missing_data_analysis'] = str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating data overview: {e}")
        
        return files
    
    def _generate_market_analysis(self, data: pd.DataFrame) -> Dict:
        """Generate market analysis visualizations"""
        files = {}
        
        try:
            # 1. VIX Analysis
            vix_cols = [col for col in data.columns if col.startswith('VIX_')]
            if vix_cols:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('VIX Market Analysis', fontsize=16, fontweight='bold')
                
                # VIX Close vs IPO Performance
                if 'VIX_Close' in data.columns and 'first_day_return' in data.columns:
                    valid_data = data[['VIX_Close', 'first_day_return']].dropna()
                    if len(valid_data) > 0:
                        axes[0, 0].scatter(valid_data['VIX_Close'], valid_data['first_day_return'], 
                                          alpha=0.6, color=self.colors['primary'])
                        axes[0, 0].set_xlabel('VIX Close')
                        axes[0, 0].set_ylabel('First Day Return (%)')
                        axes[0, 0].set_title('VIX vs IPO Performance')
                        
                        # Add trend line
                        z = np.polyfit(valid_data['VIX_Close'], valid_data['first_day_return'], 1)
                        p = np.poly1d(z)
                        axes[0, 0].plot(valid_data['VIX_Close'], p(valid_data['VIX_Close']), 
                                       "r--", alpha=0.8)
                
                # VIX Volatility vs IPO Performance
                if 'VIX_Volatility' in data.columns and 'first_day_return' in data.columns:
                    valid_data = data[['VIX_Volatility', 'first_day_return']].dropna()
                    if len(valid_data) > 0:
                        axes[0, 1].scatter(valid_data['VIX_Volatility'], valid_data['first_day_return'], 
                                          alpha=0.6, color=self.colors['secondary'])
                        axes[0, 1].set_xlabel('VIX Volatility')
                        axes[0, 1].set_ylabel('First Day Return (%)')
                        axes[0, 1].set_title('VIX Volatility vs IPO Performance')
                
                # VIX Price Range vs IPO Performance
                if 'VIX_Price_Range_Pct' in data.columns and 'first_day_return' in data.columns:
                    valid_data = data[['VIX_Price_Range_Pct', 'first_day_return']].dropna()
                    if len(valid_data) > 0:
                        axes[1, 0].scatter(valid_data['VIX_Price_Range_Pct'], valid_data['first_day_return'], 
                                          alpha=0.6, color=self.colors['success'])
                        axes[1, 0].set_xlabel('VIX Price Range (%)')
                        axes[1, 0].set_ylabel('First Day Return (%)')
                        axes[1, 0].set_title('VIX Price Range vs IPO Performance')
                
                # VIX Volume Analysis
                if 'VIX_Volume' in data.columns:
                    valid_data = data['VIX_Volume'].dropna()
                    if len(valid_data) > 0:
                        axes[1, 1].hist(valid_data, bins=30, color=self.colors['info'], alpha=0.7)
                        axes[1, 1].set_xlabel('VIX Volume')
                        axes[1, 1].set_ylabel('Frequency')
                        axes[1, 1].set_title('VIX Volume Distribution')
                
                plt.tight_layout()
                filepath = self.output_dir / "vix_market_analysis.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                files['vix_analysis'] = str(filepath)
            
            # 2. Fed Funds Rate Analysis
            if 'FEDFUNDS' in data.columns:
                plt.figure(figsize=(12, 8))
                valid_data = data[['FEDFUNDS', 'first_day_return']].dropna()
                if len(valid_data) > 0:
                    plt.scatter(valid_data['FEDFUNDS'], valid_data['first_day_return'], 
                              alpha=0.6, color=self.colors['danger'])
                    plt.xlabel('Federal Funds Rate (%)')
                    plt.ylabel('First Day Return (%)')
                    plt.title('Federal Funds Rate vs IPO Performance', fontsize=16, fontweight='bold')
                    
                    # Add trend line
                    z = np.polyfit(valid_data['FEDFUNDS'], valid_data['first_day_return'], 1)
                    p = np.poly1d(z)
                    plt.plot(valid_data['FEDFUNDS'], p(valid_data['FEDFUNDS']), "r--", alpha=0.8)
                    
                    plt.tight_layout()
                    filepath = self.output_dir / "fedfunds_analysis.png"
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    files['fedfunds_analysis'] = str(filepath)
            
            # 3. Market Conditions Summary
            if 'VIX_Close' in data.columns or 'FEDFUNDS' in data.columns:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle('Market Conditions Summary', fontsize=16, fontweight='bold')
                
                # VIX distribution
                if 'VIX_Close' in data.columns:
                    valid_vix = data['VIX_Close'].dropna()
                    if len(valid_vix) > 0:
                        axes[0].hist(valid_vix, bins=30, color=self.colors['primary'], alpha=0.7)
                        axes[0].set_xlabel('VIX Level')
                        axes[0].set_ylabel('Frequency')
                        axes[0].set_title('VIX Distribution')
                        
                        # Add mean line
                        mean_vix = valid_vix.mean()
                        axes[0].axvline(mean_vix, color='red', linestyle='--', 
                                       label=f'Mean: {mean_vix:.2f}')
                        axes[0].legend()
                
                # Fed Funds Rate distribution
                if 'FEDFUNDS' in data.columns:
                    valid_fed = data['FEDFUNDS'].dropna()
                    if len(valid_fed) > 0:
                        axes[1].hist(valid_fed, bins=30, color=self.colors['secondary'], alpha=0.7)
                        axes[1].set_xlabel('Federal Funds Rate (%)')
                        axes[1].set_ylabel('Frequency')
                        axes[1].set_title('Fed Funds Rate Distribution')
                        
                        # Add mean line
                        mean_fed = valid_fed.mean()
                        axes[1].axvline(mean_fed, color='red', linestyle='--', 
                                      label=f'Mean: {mean_fed:.2f}%')
                        axes[1].legend()
                
                plt.tight_layout()
                filepath = self.output_dir / "market_conditions_summary.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                files['market_conditions_summary'] = str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating market analysis: {e}")
        
        return files
    
    def _generate_model_performance(self, model_results: Dict) -> Dict:
        """Generate model performance visualizations"""
        files = {}
        
        try:
            # 1. Regression Model Performance
            if 'regression_results' in model_results:
                reg_results = model_results['regression_results']
                if reg_results:
                    # Create comparison chart
                    models = list(reg_results.keys())
                    train_r2 = [reg_results[model]['metrics'].get('train_r2', 0) for model in models]
                    test_r2 = [reg_results[model]['metrics'].get('test_r2', 0) for model in models]
                    
                    x = np.arange(len(models))
                    width = 0.35
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars1 = ax.bar(x - width/2, train_r2, width, label='Train R²', 
                                  color=self.colors['primary'], alpha=0.8)
                    bars2 = ax.bar(x + width/2, test_r2, width, label='Test R²', 
                                  color=self.colors['secondary'], alpha=0.8)
                    
                    ax.set_xlabel('Models')
                    ax.set_ylabel('R² Score')
                    ax.set_title('Regression Model Performance Comparison', fontsize=16, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels([reg_results[model]['name'] for model in models], rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    filepath = self.output_dir / "regression_model_performance.png"
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    files['regression_performance'] = str(filepath)
            
            # 2. Classification Model Performance
            if 'classification_results' in model_results:
                clf_results = model_results['classification_results']
                if clf_results:
                    # Create comparison chart
                    models = list(clf_results.keys())
                    train_acc = [clf_results[model]['metrics'].get('train_accuracy', 0) for model in models]
                    test_acc = [clf_results[model]['metrics'].get('test_accuracy', 0) for model in models]
                    
                    x = np.arange(len(models))
                    width = 0.35
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars1 = ax.bar(x - width/2, train_acc, width, label='Train Accuracy', 
                                  color=self.colors['success'], alpha=0.8)
                    bars2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy', 
                                  color=self.colors['info'], alpha=0.8)
                    
                    ax.set_xlabel('Models')
                    ax.set_ylabel('Accuracy Score')
                    ax.set_title('Classification Model Performance Comparison', fontsize=16, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels([clf_results[model]['name'] for model in models], rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    filepath = self.output_dir / "classification_model_performance.png"
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    files['classification_performance'] = str(filepath)
            
            # 3. Model Performance Summary
            if 'regression_results' in model_results or 'classification_results' in model_results:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
                
                # Best regression model
                if 'regression_results' in model_results and model_results['regression_results']:
                    best_reg = max(model_results['regression_results'].items(), 
                                 key=lambda x: x[1]['metrics'].get('test_r2', 0))
                    axes[0].pie([best_reg[1]['metrics'].get('test_r2', 0), 
                                1 - best_reg[1]['metrics'].get('test_r2', 0)], 
                               labels=[f"R²: {best_reg[1]['metrics'].get('test_r2', 0):.3f}", "Remaining"],
                               colors=[self.colors['primary'], self.colors['light']],
                               autopct='%1.1f%%')
                    axes[0].set_title(f'Best Regression: {best_reg[1]["name"]}')
                
                # Best classification model
                if 'classification_results' in model_results and model_results['classification_results']:
                    best_clf = max(model_results['classification_results'].items(), 
                                 key=lambda x: x[1]['metrics'].get('test_accuracy', 0))
                    axes[1].pie([best_clf[1]['metrics'].get('test_accuracy', 0), 
                                1 - best_clf[1]['metrics'].get('test_accuracy', 0)], 
                               labels=[f"Accuracy: {best_clf[1]['metrics'].get('test_accuracy', 0):.3f}", "Remaining"],
                               colors=[self.colors['success'], self.colors['light']],
                               autopct='%1.1f%%')
                    axes[1].set_title(f'Best Classification: {best_clf[1]["name"]}')
                
                plt.tight_layout()
                filepath = self.output_dir / "model_performance_summary.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                files['model_performance_summary'] = str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating model performance: {e}")
        
        return files
    
    def _generate_feature_analysis(self, feature_importance: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """Generate feature analysis visualizations"""
        files = {}
        
        try:
            if feature_importance.empty:
                return files
            
            # 1. Feature Importance Bar Chart
            plt.figure(figsize=(14, 10))
            top_features = feature_importance.head(20)  # Top 20 features
            
            plt.barh(range(len(top_features)), top_features['importance'], 
                    color=self.colors['primary'], alpha=0.8)
            plt.yticks(range(len(top_features)), [f'Feature {int(f)}' for f in top_features['feature']])
            plt.xlabel('Importance Score')
            plt.title('Top 20 Feature Importance Scores', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filepath = self.output_dir / "feature_importance.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            files['feature_importance'] = str(filepath)
            
            # 2. Feature Importance Distribution
            plt.figure(figsize=(12, 8))
            plt.hist(feature_importance['importance'], bins=30, color=self.colors['secondary'], alpha=0.7)
            plt.xlabel('Importance Score')
            plt.ylabel('Frequency')
            plt.title('Feature Importance Distribution', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add mean line
            mean_importance = feature_importance['importance'].mean()
            plt.axvline(mean_importance, color='red', linestyle='--', 
                       label=f'Mean: {mean_importance:.4f}')
            plt.legend()
            
            plt.tight_layout()
            filepath = self.output_dir / "feature_importance_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            files['feature_importance_distribution'] = str(filepath)
            
            # 3. Feature Correlation with Target
            if 'close_price_target' in data.columns:
                numeric_features = data.select_dtypes(include=[np.number]).columns
                target_correlations = []
                feature_names = []
                
                for col in numeric_features:
                    if col != 'close_price_target' and col != 'price_direction':
                        corr = data[col].corr(data['close_price_target'])
                        if not pd.isna(corr):
                            target_correlations.append(abs(corr))
                            feature_names.append(col)
                
                if target_correlations:
                    # Sort by correlation
                    sorted_data = sorted(zip(feature_names, target_correlations), 
                                       key=lambda x: x[1], reverse=True)
                    top_corr_features = sorted_data[:20]
                    
                    feature_names_sorted = [x[0] for x in top_corr_features]
                    correlations_sorted = [x[1] for x in top_corr_features]
                    
                    plt.figure(figsize=(14, 10))
                    plt.barh(range(len(feature_names_sorted)), correlations_sorted, 
                            color=self.colors['success'], alpha=0.8)
                    plt.yticks(range(len(feature_names_sorted)), feature_names_sorted)
                    plt.xlabel('Absolute Correlation with Target')
                    plt.title('Top 20 Features by Target Correlation', fontsize=16, fontweight='bold')
                    plt.gca().invert_yaxis()
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    filepath = self.output_dir / "feature_target_correlation.png"
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    files['feature_target_correlation'] = str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating feature analysis: {e}")
        
        return files
    
    def _generate_interactive_dashboard(self, data: pd.DataFrame, model_results: Dict, 
                                      feature_importance: pd.DataFrame = None) -> str:
        """Generate interactive Plotly dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('IPO Performance Over Time', 'Price vs Return Scatter',
                              'VIX vs Performance', 'Model Performance Comparison',
                              'Feature Importance', 'Market Conditions'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # 1. IPO Performance Over Time
            if 'Date' in data.columns and 'first_day_return' in data.columns:
                valid_data = data[['Date', 'first_day_return']].dropna()
                if len(valid_data) > 0:
                    fig.add_trace(
                        go.Scatter(x=valid_data['Date'], y=valid_data['first_day_return'],
                                 mode='markers', name='IPO Returns',
                                 marker=dict(color=valid_data['first_day_return'], 
                                           colorscale='RdYlGn', showscale=True)),
                        row=1, col=1
                    )
            
            # 2. Price vs Return Scatter
            if 'Price' in data.columns and 'first_day_return' in data.columns:
                valid_data = data[['Price', 'first_day_return']].dropna()
                if len(valid_data) > 0:
                    fig.add_trace(
                        go.Scatter(x=valid_data['Price'], y=valid_data['first_day_return'],
                                 mode='markers', name='Price vs Return',
                                 marker=dict(color=self.colors['primary'], opacity=0.7)),
                        row=1, col=2
                    )
            
            # 3. VIX vs Performance
            if 'VIX_Close' in data.columns and 'first_day_return' in data.columns:
                valid_data = data[['VIX_Close', 'first_day_return']].dropna()
                if len(valid_data) > 0:
                    fig.add_trace(
                        go.Scatter(x=valid_data['VIX_Close'], y=valid_data['first_day_return'],
                                 mode='markers', name='VIX vs Performance',
                                 marker=dict(color=self.colors['secondary'], opacity=0.7)),
                        row=2, col=1
                    )
            
            # 4. Model Performance Comparison
            if 'regression_results' in model_results:
                reg_results = model_results['regression_results']
                if reg_results:
                    models = list(reg_results.keys())
                    test_r2 = [reg_results[model]['metrics'].get('test_r2', 0) for model in models]
                    
                    fig.add_trace(
                        go.Bar(x=[reg_results[model]['name'] for model in models], 
                              y=test_r2, name='Test R²',
                              marker_color=self.colors['success']),
                        row=2, col=2
                    )
            
            # 5. Feature Importance
            if feature_importance is not None and not feature_importance.empty:
                top_features = feature_importance.head(15)
                fig.add_trace(
                    go.Bar(x=[f'Feature {int(f)}' for f in top_features['feature']], 
                          y=top_features['importance'], name='Feature Importance',
                          marker_color=self.colors['info']),
                    row=3, col=1
                )
            
            # 6. Market Conditions
            if 'VIX_Close' in data.columns:
                valid_vix = data['VIX_Close'].dropna()
                if len(valid_vix) > 0:
                    fig.add_trace(
                        go.Scatter(x=valid_vix.index, y=valid_vix.values,
                                 mode='lines', name='VIX Trend',
                                 line=dict(color=self.colors['warning'])),
                        row=3, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title_text="IPO Analysis Interactive Dashboard",
                showlegend=True,
                height=1200,
                width=1600
            )
            
            # Save interactive HTML
            filepath = self.output_dir / "interactive_dashboard.html"
            fig.write_html(str(filepath))
            
            logger.info(f"Interactive dashboard saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating interactive dashboard: {e}")
            return ""
    
    def _generate_summary_report(self, data: pd.DataFrame, model_results: Dict, 
                                generated_files: Dict) -> str:
        """Generate summary report with all visualizations"""
        try:
            report_content = f"""
# IPO Analysis Pipeline - Comprehensive Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides comprehensive analysis of IPO performance using machine learning models and market data integration.

## Data Overview
- **Total IPO Records**: {len(data):,}
- **Features Available**: {len(data.columns):,}
- **Date Range**: {data['Date'].min() if 'Date' in data.columns else 'N/A'} to {data['Date'].max() if 'Date' in data.columns else 'N/A'}

## Key Findings

### Market Performance
- **Average First Day Return**: {data['first_day_return'].mean():.2f}% (if available)
- **IPO Price Range**: ${data['Price'].min():.2f} - ${data['Price'].max():.2f} (if available)
- **Market Volatility**: VIX data integrated for {data['VIX_Close'].notna().sum():,} records (if available)

### Model Performance
"""
            
            # Add model performance summary
            if 'regression_results' in model_results:
                reg_results = model_results['regression_results']
                if reg_results:
                    best_reg = max(reg_results.items(), 
                                 key=lambda x: x[1]['metrics'].get('test_r2', 0))
                    report_content += f"""
#### Regression Models
- **Best Model**: {best_reg[1]['name']}
- **Test R² Score**: {best_reg[1]['metrics'].get('test_r2', 0):.4f}
- **Total Models Trained**: {len(reg_results)}
"""
            
            if 'classification_results' in model_results:
                clf_results = model_results['classification_results']
                if clf_results:
                    best_clf = max(clf_results.items(), 
                                 key=lambda x: x[1]['metrics'].get('test_accuracy', 0))
                    report_content += f"""
#### Classification Models
- **Best Model**: {best_clf[1]['name']}
- **Test Accuracy**: {best_clf[1]['metrics'].get('test_accuracy', 0):.4f}
- **Total Models Trained**: {len(clf_results)}
"""
            
            report_content += f"""
## Generated Visualizations
The following visualizations have been created to support this analysis:

"""
            
            # List all generated files
            for viz_type, filepath in generated_files.items():
                if viz_type != 'summary_report':
                    report_content += f"- **{viz_type.replace('_', ' ').title()}**: {filepath}\n"
            
            report_content += f"""
## Technical Details
- **Data Sources**: IPO details, VIX market data, Federal Funds Rate, SEC filings
- **Feature Engineering**: Advanced feature creation including market indicators and NLP features
- **Model Types**: Regression (price prediction) and Classification (direction prediction)
- **Evaluation Metrics**: R², RMSE, MAE for regression; Accuracy for classification

## Recommendations
1. **Model Selection**: Use the best performing model identified above for predictions
2. **Feature Importance**: Focus on the most important features for model improvement
3. **Market Timing**: Consider VIX and Fed Funds Rate for IPO timing decisions
4. **Continuous Monitoring**: Regularly retrain models with new data

## Files Generated
All visualizations and reports have been saved to the `{self.output_dir}` directory.

---
*Report generated automatically by IPO Analysis ML Pipeline*
"""
            
            # Save report
            filepath = self.output_dir / "comprehensive_report.md"
            with open(filepath, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Summary report saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return ""
    
    def generate_quick_charts(self, data: pd.DataFrame, save_dir: str = None) -> List[str]:
        """
        Generate quick charts for immediate analysis
        
        Args:
            data: DataFrame to visualize
            save_dir: Directory to save charts (optional)
            
        Returns:
            List of saved file paths
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        saved_files = []
        
        try:
            # Quick price distribution
            if 'Price' in data.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(data['Price'].dropna(), bins=30, color=self.colors['primary'], alpha=0.7)
                plt.title('IPO Price Distribution')
                plt.xlabel('IPO Price ($)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                filepath = Path(save_dir) / "quick_price_distribution.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                saved_files.append(str(filepath))
            
            # Quick return distribution
            if 'first_day_return' in data.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(data['first_day_return'].dropna(), bins=30, color=self.colors['success'], alpha=0.7)
                plt.title('First Day Return Distribution')
                plt.xlabel('First Day Return (%)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                filepath = Path(save_dir) / "quick_return_distribution.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                saved_files.append(str(filepath))
            
            # Quick correlation matrix
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = data[numeric_cols].corr()
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5)
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                
                filepath = Path(save_dir) / "quick_correlation_matrix.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                saved_files.append(str(filepath))
            
            logger.info(f"Generated {len(saved_files)} quick charts")
            
        except Exception as e:
            logger.error(f"Error generating quick charts: {e}")
        
        return saved_files
