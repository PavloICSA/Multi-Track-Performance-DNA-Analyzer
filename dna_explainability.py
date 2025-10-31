#!/usr/bin/env python3
"""
DNA Explainability Engine
Provides feature importance and interpretability for driver archetype classification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class DNAExplainability:
    """
    Explainability engine for driver DNA analysis.
    Provides feature importance and interpretability for archetype classification.
    """
    
    def __init__(self):
        """Initialize explainability engine"""
        self.feature_names = [
            'speed_vs_consistency_ratio',
            'track_adaptability',
            'consistency_index',
            'performance_variance',
            'speed_consistency',
            'technical_track_performance',
            'high_speed_track_performance',
            'mixed_track_performance',
            'sector_balance_score'
        ]
        
        self.feature_display_names = {
            'speed_vs_consistency_ratio': 'Speed/Consistency Ratio',
            'track_adaptability': 'Track Adaptability',
            'consistency_index': 'Consistency Index',
            'performance_variance': 'Performance Variance',
            'speed_consistency': 'Speed Consistency',
            'technical_track_performance': 'Technical Track Performance',
            'high_speed_track_performance': 'High-Speed Track Performance',
            'mixed_track_performance': 'Mixed Track Performance',
            'sector_balance_score': 'Sector Balance Score'
        }
        
        self.feature_descriptions = {
            'speed_vs_consistency_ratio': 'Ratio of average speed to lap time variance. Higher values indicate fast and consistent driving.',
            'track_adaptability': 'Ability to maintain consistent performance across different tracks. Higher is better.',
            'consistency_index': 'Lap-to-lap consistency score. Higher values mean more predictable lap times.',
            'performance_variance': 'Variation in performance across tracks. Lower values indicate more consistent performance.',
            'speed_consistency': 'Variation in speed across tracks. Lower values indicate more stable speed management.',
            'technical_track_performance': 'Relative performance on technical tracks (Barber, Sebring, Sonoma).',
            'high_speed_track_performance': 'Relative performance on high-speed tracks (Road America).',
            'mixed_track_performance': 'Relative performance on mixed tracks (COTA, VIR).',
            'sector_balance_score': 'How evenly distributed sector times are. Higher values indicate balanced performance.'
        }
    
    def calculate_feature_importance_by_archetype(self, dna_features: pd.DataFrame, 
                                                   archetypes: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Calculate feature importance for each archetype using statistical analysis.
        
        Args:
            dna_features: DataFrame with DNA features
            archetypes: Series with archetype labels
            
        Returns:
            Dictionary mapping archetype to feature importance DataFrame
        """
        if dna_features.empty or archetypes.empty:
            return {}
        
        # Combine features and archetypes
        data = dna_features.copy()
        data['archetype'] = archetypes.values
        
        importance_by_archetype = {}
        
        for archetype in archetypes.unique():
            archetype_data = data[data['archetype'] == archetype]
            other_data = data[data['archetype'] != archetype]
            
            if len(archetype_data) == 0 or len(other_data) == 0:
                continue
            
            feature_importance = []
            
            for feature in self.feature_names:
                if feature not in dna_features.columns:
                    continue
                
                # Calculate mean difference (effect size)
                archetype_mean = archetype_data[feature].mean()
                other_mean = other_data[feature].mean()
                mean_diff = archetype_mean - other_mean
                
                # Calculate normalized importance (0-100 scale)
                # Use absolute difference normalized by overall standard deviation
                overall_std = data[feature].std()
                if overall_std > 0:
                    importance_score = abs(mean_diff) / overall_std * 100
                else:
                    importance_score = 0
                
                # Calculate direction (positive = higher for this archetype)
                direction = 'higher' if mean_diff > 0 else 'lower'
                
                feature_importance.append({
                    'feature': feature,
                    'display_name': self.feature_display_names.get(feature, feature),
                    'importance': importance_score,
                    'archetype_mean': archetype_mean,
                    'other_mean': other_mean,
                    'difference': mean_diff,
                    'direction': direction,
                    'description': self.feature_descriptions.get(feature, '')
                })
            
            # Sort by importance
            importance_df = pd.DataFrame(feature_importance)
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            importance_by_archetype[archetype] = importance_df
        
        return importance_by_archetype
    
    def get_top_features_for_archetype(self, archetype: str, 
                                       importance_data: Dict[str, pd.DataFrame],
                                       top_n: int = 5) -> pd.DataFrame:
        """
        Get top N most important features for a specific archetype.
        
        Args:
            archetype: Archetype name
            importance_data: Dictionary from calculate_feature_importance_by_archetype
            top_n: Number of top features to return
            
        Returns:
            DataFrame with top features
        """
        if archetype not in importance_data:
            return pd.DataFrame()
        
        return importance_data[archetype].head(top_n)
    
    def explain_driver_archetype(self, driver_dna: pd.Series, archetype: str,
                                 importance_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Explain why a driver was classified into a specific archetype.
        
        Args:
            driver_dna: Series with driver's DNA features
            archetype: Driver's archetype
            importance_data: Dictionary from calculate_feature_importance_by_archetype
            
        Returns:
            Dictionary with explanation details
        """
        if archetype not in importance_data:
            return {'error': 'Archetype not found in importance data'}
        
        top_features = self.get_top_features_for_archetype(archetype, importance_data, top_n=3)
        
        explanation = {
            'archetype': archetype,
            'key_factors': [],
            'driver_values': {}
        }
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            if feature in driver_dna.index:
                driver_value = driver_dna[feature]
                explanation['key_factors'].append({
                    'feature': row['display_name'],
                    'importance': row['importance'],
                    'driver_value': driver_value,
                    'archetype_mean': row['archetype_mean'],
                    'direction': row['direction'],
                    'description': row['description']
                })
                explanation['driver_values'][feature] = driver_value
        
        return explanation
    
    def generate_archetype_summary(self, importance_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Generate human-readable summaries for each archetype.
        
        Args:
            importance_data: Dictionary from calculate_feature_importance_by_archetype
            
        Returns:
            Dictionary mapping archetype to summary text
        """
        summaries = {}
        
        for archetype, importance_df in importance_data.items():
            if importance_df.empty:
                continue
            
            top_3 = importance_df.head(3)
            
            # Build summary text
            summary_parts = [f"**{archetype}** drivers are characterized by:"]
            
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                feature_name = row['display_name']
                direction = row['direction']
                summary_parts.append(f"{i}. **{direction.capitalize()} {feature_name}** - {row['description']}")
            
            summaries[archetype] = '\n'.join(summary_parts)
        
        return summaries
    
    def calculate_feature_correlations(self, dna_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between DNA features.
        
        Args:
            dna_features: DataFrame with DNA features
            
        Returns:
            Correlation matrix DataFrame
        """
        feature_cols = [col for col in self.feature_names if col in dna_features.columns]
        
        if not feature_cols:
            return pd.DataFrame()
        
        return dna_features[feature_cols].corr()
    
    def get_feature_statistics(self, dna_features: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistical summary of all DNA features.
        
        Args:
            dna_features: DataFrame with DNA features
            
        Returns:
            DataFrame with feature statistics
        """
        feature_cols = [col for col in self.feature_names if col in dna_features.columns]
        
        if not feature_cols:
            return pd.DataFrame()
        
        stats = dna_features[feature_cols].describe().T
        stats['display_name'] = stats.index.map(self.feature_display_names)
        stats = stats[['display_name', 'mean', 'std', 'min', 'max']]
        
        return stats.round(3)


def main():
    """Example usage of DNAExplainability"""
    print("🔍 DNA Explainability Engine")
    print("=" * 50)
    
    # Example with synthetic data
    np.random.seed(42)
    
    # Create sample DNA features
    n_drivers = 20
    dna_features = pd.DataFrame({
        'driver_id': range(1, n_drivers + 1),
        'speed_vs_consistency_ratio': np.random.uniform(5, 15, n_drivers),
        'track_adaptability': np.random.uniform(0.5, 2.0, n_drivers),
        'consistency_index': np.random.uniform(0.05, 0.15, n_drivers),
        'performance_variance': np.random.uniform(0.005, 0.025, n_drivers),
        'speed_consistency': np.random.uniform(0.02, 0.08, n_drivers),
        'technical_track_performance': np.random.uniform(0.9, 1.1, n_drivers),
        'high_speed_track_performance': np.random.uniform(0.9, 1.1, n_drivers),
        'mixed_track_performance': np.random.uniform(0.9, 1.1, n_drivers),
        'sector_balance_score': np.random.uniform(0.8, 1.2, n_drivers)
    })
    
    # Create sample archetypes
    archetypes = pd.Series(['Speed Demon'] * 5 + ['Consistency Master'] * 5 + 
                          ['Track Specialist'] * 5 + ['Balanced Racer'] * 5)
    
    # Initialize explainability engine
    explainer = DNAExplainability()
    
    # Calculate feature importance
    importance = explainer.calculate_feature_importance_by_archetype(dna_features, archetypes)
    
    print(f"✅ Calculated feature importance for {len(importance)} archetypes")
    
    # Display results
    for archetype, importance_df in importance.items():
        print(f"\n🏁 {archetype} - Top 3 Features:")
        top_3 = importance_df.head(3)
        for _, row in top_3.iterrows():
            print(f"  • {row['display_name']}: {row['importance']:.1f}% importance ({row['direction']})")
    
    # Generate summaries
    summaries = explainer.generate_archetype_summary(importance)
    print("\n📋 Archetype Summaries:")
    for archetype, summary in summaries.items():
        print(f"\n{summary}")


if __name__ == "__main__":
    main()
