#!/usr/bin/env python3
"""
Multi-Track Performance DNA Analyzer
Creates driver "performance fingerprints" across all 6 tracks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, Any, Union, List
import warnings
warnings.filterwarnings('ignore')

class PerformanceDNAAnalyzer:
    def __init__(self, use_pretrained: bool = False, model_dir: str = None):
        """
        Initialize Performance DNA Analyzer
        
        Args:
            use_pretrained: If True, use ML model for inference; if False, use original logic
            model_dir: Path to model artifacts directory (uses 'models/latest' if None)
        """
        self.use_pretrained = use_pretrained
        self.tracks = ['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']
        self.track_data = {}
        self.driver_profiles = {}
        self.scaler = StandardScaler()
        
        # Initialize model-based components if requested
        if use_pretrained:
            try:
                from dna_model_inference import DNAModelInference
                from dna_feature_engineering import DNAFeatureEngineering
                
                self.inference_engine = DNAModelInference(model_dir=model_dir)
                self.feature_engineering = DNAFeatureEngineering()
                print("✅ Initialized with pre-trained model")
            except Exception as e:
                print(f"⚠️ Failed to load pre-trained model: {e}")
                print("   Falling back to original implementation")
                self.use_pretrained = False
                self.inference_engine = None
                self.feature_engineering = None
        else:
            self.inference_engine = None
            self.feature_engineering = None
        
    def load_track_data(self, data_source=None):
        """
        Load and process data from all tracks or user-provided data
        
        Args:
            data_source: Can be:
                - None: Load from default track directories (original behavior)
                - str: Path to directory containing track data
                - pd.DataFrame: User-provided DataFrame with racing data
                - List[pd.DataFrame]: List of DataFrames from multiple sources
        
        Returns:
            self for method chaining
        """
        # Handle DataFrame input (for model-based inference)
        if isinstance(data_source, pd.DataFrame):
            print("🔄 Loading user-provided DataFrame...")
            self._load_from_dataframe(data_source)
            print("✅ Data loading complete!")
            return self
        
        # Handle list of DataFrames
        if isinstance(data_source, list) and all(isinstance(df, pd.DataFrame) for df in data_source):
            print("🔄 Loading user-provided DataFrames...")
            combined_df = pd.concat(data_source, ignore_index=True)
            self._load_from_dataframe(combined_df)
            print("✅ Data loading complete!")
            return self
        
        # Handle directory path input
        if isinstance(data_source, str):
            data_path = Path(data_source)
            if not data_path.exists():
                raise ValueError(f"Data directory not found: {data_source}")
            print(f"🔄 Loading data from directory: {data_source}...")
            # Use original loading logic with custom path
            self._load_from_directory(data_path)
            print("✅ Data loading complete!")
            return self
        
        # Default behavior: load from track directories
        print("🔄 Loading data from all tracks...")
        
        for track in self.tracks:
            print(f"   📍 Processing {track}...")
            track_path = Path(track)
            
            track_info = {
                'sector_data': [],
                'best_laps': [],
                'weather': [],
                'results': []
            }
            
            # Handle different directory structures
            if track == 'barber':
                race_paths = [track_path]  # Files are directly in barber folder
            else:
                race_paths = [track_path / 'Race 1', track_path / 'Race 2']
            
            for race_path in race_paths:
                if not race_path.exists():
                    continue
                    
                # Load sector analysis data
                sector_files = list(race_path.glob('*AnalysisEnduranceWithSections*.CSV'))
                for file in sector_files:
                    try:
                        # Try different delimiters
                        for delimiter in [';', ',', '\t']:
                            try:
                                df = pd.read_csv(file, delimiter=delimiter)
                                if len(df.columns) > 5:  # Valid data
                                    # Clean column names (remove leading/trailing spaces)
                                    df.columns = df.columns.str.strip()
                                    df['track'] = track
                                    df['race'] = race_path.name if race_path.name.startswith('Race') else 'R1'
                                    track_info['sector_data'].append(df)
                                    break
                            except:
                                continue
                    except Exception as e:
                        print(f"      ⚠️ Error loading {file.name}: {e}")
                
                # Load best laps data
                best_lap_files = list(race_path.glob('*Best 10 Laps*.CSV'))
                for file in best_lap_files:
                    try:
                        for delimiter in [';', ',', '\t']:
                            try:
                                df = pd.read_csv(file, delimiter=delimiter)
                                if len(df.columns) > 5:
                                    # Clean column names
                                    df.columns = df.columns.str.strip()
                                    df['track'] = track
                                    df['race'] = race_path.name if race_path.name.startswith('Race') else 'R1'
                                    track_info['best_laps'].append(df)
                                    break
                            except:
                                continue
                    except Exception as e:
                        print(f"      ⚠️ Error loading {file.name}: {e}")
                
                # Load weather data
                weather_files = list(race_path.glob('*Weather*.CSV'))
                for file in weather_files:
                    try:
                        for delimiter in [';', ',', '\t']:
                            try:
                                df = pd.read_csv(file, delimiter=delimiter)
                                if len(df.columns) > 3:
                                    # Clean column names
                                    df.columns = df.columns.str.strip()
                                    df['track'] = track
                                    df['race'] = race_path.name if race_path.name.startswith('Race') else 'R1'
                                    track_info['weather'].append(df)
                                    break
                            except:
                                continue
                    except Exception as e:
                        print(f"      ⚠️ Error loading {file.name}: {e}")
            
            self.track_data[track] = track_info
            
        print("✅ Data loading complete!")
        return self
    
    def _load_from_dataframe(self, df: pd.DataFrame):
        """
        Load data from a user-provided DataFrame
        
        Args:
            df: DataFrame with columns like NUMBER, LAP_TIME, S1, S2, S3, KPH, track
        """
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Ensure track column exists
        if 'track' not in df.columns:
            # If no track column, assign a default track name
            df['track'] = 'user_track'
        
        # Group by track
        tracks = df['track'].unique()
        
        for track in tracks:
            track_df = df[df['track'] == track].copy()
            
            # Store in track_data structure
            if track not in self.track_data:
                self.track_data[track] = {
                    'sector_data': [],
                    'best_laps': [],
                    'weather': [],
                    'results': []
                }
            
            self.track_data[track]['sector_data'].append(track_df)
        
        # Update tracks list to include user tracks
        self.tracks = list(set(self.tracks + list(tracks)))
    
    def _load_from_directory(self, directory_path: Path):
        """
        Load data from a custom directory path
        
        Args:
            directory_path: Path to directory containing track data
        """
        # Similar to default loading but from custom path
        # This is a simplified version - can be expanded as needed
        track_name = directory_path.name
        
        track_info = {
            'sector_data': [],
            'best_laps': [],
            'weather': [],
            'results': []
        }
        
        # Load sector analysis data
        sector_files = list(directory_path.glob('*AnalysisEnduranceWithSections*.CSV'))
        for file in sector_files:
            try:
                for delimiter in [';', ',', '\t']:
                    try:
                        df = pd.read_csv(file, delimiter=delimiter)
                        if len(df.columns) > 5:
                            df.columns = df.columns.str.strip()
                            df['track'] = track_name
                            track_info['sector_data'].append(df)
                            break
                    except:
                        continue
            except Exception as e:
                print(f"   ⚠️ Error loading {file.name}: {e}")
        
        self.track_data[track_name] = track_info
        if track_name not in self.tracks:
            self.tracks.append(track_name)
    
    def analyze_sector_performance(self):
        """Analyze sector performance across all tracks"""
        print("🔍 Analyzing sector performance...")
        
        all_sector_data = []
        for track, data in self.track_data.items():
            for df in data['sector_data']:
                if not df.empty and 'NUMBER' in df.columns:
                    all_sector_data.append(df)
        
        if not all_sector_data:
            print("❌ No sector data found")
            return self
            
        combined_sectors = pd.concat(all_sector_data, ignore_index=True)
        
        # Clean and process sector times
        sector_cols = ['S1', 'S2', 'S3']
        time_cols = ['LAP_TIME']
        
        # Convert time strings to seconds if needed
        def convert_time_to_seconds(time_str):
            """Convert MM:SS.mmm format to seconds"""
            if pd.isna(time_str) or time_str == '':
                return np.nan
            try:
                if isinstance(time_str, str) and ':' in time_str:
                    parts = time_str.split(':')
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    return float(time_str)
            except:
                return np.nan
        
        for col in sector_cols + time_cols:
            if col in combined_sectors.columns:
                combined_sectors[col] = combined_sectors[col].apply(convert_time_to_seconds)
        
        # Calculate driver performance metrics per track
        driver_track_performance = combined_sectors.groupby(['NUMBER', 'track']).agg({
            'LAP_TIME': ['mean', 'std', 'min', 'count'],
            'S1': ['mean', 'std', 'min'],
            'S2': ['mean', 'std', 'min'], 
            'S3': ['mean', 'std', 'min'],
            'KPH': ['mean', 'max']
        }).round(3)
        
        driver_track_performance.columns = ['_'.join(col).strip() for col in driver_track_performance.columns]
        driver_track_performance = driver_track_performance.reset_index()
        
        self.sector_analysis = driver_track_performance
        print(f"✅ Analyzed {len(driver_track_performance)} driver-track combinations")
        return self
    
    def create_driver_dna_profiles(self):
        """
        Create comprehensive DNA profiles for each driver
        
        Routes to inference engine if use_pretrained=True,
        otherwise uses original calculation logic
        
        Returns:
            self for method chaining
        """
        print("🧬 Creating driver DNA profiles...")
        
        # Route to inference engine if using pre-trained model
        if self.use_pretrained and self.inference_engine is not None:
            return self._create_profiles_with_model()
        
        # Original implementation
        if not hasattr(self, 'sector_analysis'):
            print("❌ No sector analysis available")
            return self
        
        # Create feature matrix for each driver across all tracks
        drivers = self.sector_analysis['NUMBER'].unique()
        
        for driver in drivers:
            driver_data = self.sector_analysis[self.sector_analysis['NUMBER'] == driver]
            
            profile = {
                'driver_id': driver,
                'tracks_raced': list(driver_data['track'].unique()),
                'total_races': len(driver_data),
                'performance_metrics': {}
            }
            
            # Track-specific performance
            for track in driver_data['track'].unique():
                track_data = driver_data[driver_data['track'] == track].iloc[0]
                
                profile['performance_metrics'][track] = {
                    'avg_lap_time': track_data.get('LAP_TIME_mean', 0),
                    'consistency': 1 / (track_data.get('LAP_TIME_std', 1) + 0.001),  # Higher = more consistent
                    'best_lap': track_data.get('LAP_TIME_min', 0),
                    'sector_balance': {
                        'S1_avg': track_data.get('S1_mean', 0),
                        'S2_avg': track_data.get('S2_mean', 0),
                        'S3_avg': track_data.get('S3_mean', 0)
                    },
                    'speed_profile': {
                        'avg_speed': track_data.get('KPH_mean', 0),
                        'top_speed': track_data.get('KPH_max', 0)
                    }
                }
            
            # Calculate cross-track DNA characteristics
            profile['dna_signature'] = self._calculate_dna_signature(profile)
            
            self.driver_profiles[driver] = profile
        
        print(f"✅ Created DNA profiles for {len(self.driver_profiles)} drivers")
        return self
    
    def _create_profiles_with_model(self):
        """
        Create driver profiles using pre-trained model inference
        
        Returns:
            self for method chaining
        """
        try:
            # Collect all sector data from track_data
            all_sector_data = []
            for track, data in self.track_data.items():
                for df in data['sector_data']:
                    if not df.empty:
                        all_sector_data.append(df)
            
            if not all_sector_data:
                print("❌ No sector data available for inference")
                return self
            
            # Combine all data
            combined_data = pd.concat(all_sector_data, ignore_index=True)
            
            # Use inference engine to create profiles
            self.driver_profiles = self.inference_engine.create_driver_profiles(combined_data)
            
            print(f"✅ Created DNA profiles for {len(self.driver_profiles)} drivers using ML model")
            
        except Exception as e:
            print(f"⚠️ Model inference failed: {e}")
            print("   Falling back to original implementation")
            self.use_pretrained = False
            # Retry with original implementation
            return self.create_driver_dna_profiles()
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model version, metrics, and status
            Returns None if not using pre-trained model
        """
        if not self.use_pretrained or self.inference_engine is None:
            return {
                'status': 'not_using_model',
                'mode': 'original_implementation',
                'message': 'Using original calculation logic'
            }
        
        try:
            model_info = self.inference_engine.get_model_info()
            
            # Format for UI display
            formatted_info = {
                'status': model_info.get('status', 'unknown'),
                'mode': 'model_based',
                'version': model_info.get('version', 'unknown'),
                'training_date': model_info.get('training_date', 'unknown'),
                'model_type': model_info.get('model_type', 'unknown'),
                'framework': model_info.get('framework', 'unknown'),
                'metrics': model_info.get('validation_metrics', {}),
                'reliability_score': model_info.get('validation_metrics', {}).get('overall_reliability', None),
                'archetype_classes': model_info.get('archetype_classes', []),
                'feature_count': len(model_info.get('feature_names', []))
            }
            
            return formatted_info
            
        except Exception as e:
            return {
                'status': 'error',
                'mode': 'model_based',
                'error': str(e),
                'message': 'Failed to retrieve model information'
            }
    
    def _calculate_dna_signature(self, profile):
        """Calculate unique DNA signature characteristics"""
        tracks = profile['tracks_raced']
        if len(tracks) < 2:
            return {'insufficient_data': True}
        
        # Extract performance metrics across tracks
        lap_times = [profile['performance_metrics'][track]['avg_lap_time'] 
                    for track in tracks if profile['performance_metrics'][track]['avg_lap_time'] > 0]
        
        consistencies = [profile['performance_metrics'][track]['consistency'] 
                        for track in tracks]
        
        speeds = [profile['performance_metrics'][track]['speed_profile']['avg_speed'] 
                 for track in tracks if profile['performance_metrics'][track]['speed_profile']['avg_speed'] > 0]
        
        if not lap_times or not speeds:
            return {'insufficient_data': True}
        
        # Calculate DNA characteristics
        dna = {
            'speed_vs_consistency_ratio': np.mean(speeds) / (np.std(lap_times) + 0.001),
            'track_adaptability': 1 / (np.std(lap_times) / np.mean(lap_times) + 0.001),
            'consistency_index': np.mean(consistencies),
            'performance_variance': np.std(lap_times) / np.mean(lap_times),
            'speed_consistency': np.std(speeds) / np.mean(speeds) if speeds else 0,
            'track_specialization': self._calculate_track_specialization(profile)
        }
        
        return dna
    
    def _calculate_track_specialization(self, profile):
        """Determine if driver specializes in certain track types"""
        # Simplified track categorization (would need more track data for accurate classification)
        track_categories = {
            'barber': 'technical',
            'COTA': 'mixed',
            'Road America': 'high_speed',
            'Sebring': 'technical', 
            'Sonoma': 'technical',
            'VIR': 'mixed'
        }
        
        performance_by_category = {}
        for track in profile['tracks_raced']:
            category = track_categories.get(track, 'unknown')
            if category not in performance_by_category:
                performance_by_category[category] = []
            
            lap_time = profile['performance_metrics'][track]['avg_lap_time']
            if lap_time > 0:
                performance_by_category[category].append(lap_time)
        
        # Calculate relative performance in each category
        specialization = {}
        for category, times in performance_by_category.items():
            if times:
                specialization[f'{category}_performance'] = np.mean(times)
        
        return specialization
    
    def visualize_driver_dna(self, driver_id=None):
        """Create comprehensive DNA visualization"""
        print("📊 Creating DNA visualizations...")
        
        if not self.driver_profiles:
            print("❌ No driver profiles available")
            return
        
        # If specific driver requested, show detailed view
        if driver_id and driver_id in self.driver_profiles:
            self._create_individual_dna_chart(driver_id)
        else:
            # Show comparative analysis of all drivers
            self._create_comparative_dna_analysis()
    
    def _create_individual_dna_chart(self, driver_id):
        """Create detailed DNA chart for individual driver"""
        profile = self.driver_profiles[driver_id]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Driver {driver_id} - Track Performance',
                'Sector Time Distribution',
                'DNA Signature Radar',
                'Performance Consistency'
            ),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatterpolar"}, {"type": "scatter"}]]
        )
        
        # Track performance comparison
        tracks = list(profile['performance_metrics'].keys())
        lap_times = [profile['performance_metrics'][track]['avg_lap_time'] for track in tracks]
        
        fig.add_trace(
            go.Bar(x=tracks, y=lap_times, name='Avg Lap Time'),
            row=1, col=1
        )
        
        # DNA Signature Radar Chart
        dna = profile['dna_signature']
        if not dna.get('insufficient_data', False):
            radar_categories = list(dna.keys())[:6]  # Top 6 DNA characteristics
            radar_values = [dna[cat] for cat in radar_categories]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_values,
                    theta=radar_categories,
                    fill='toself',
                    name=f'Driver {driver_id} DNA'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'Performance DNA Profile - Driver {driver_id}',
            height=800
        )
        
        fig.show()
    
    def _create_comparative_dna_analysis(self):
        """Create comparative analysis across all drivers"""
        # Extract DNA features for clustering
        dna_features = []
        driver_ids = []
        
        for driver_id, profile in self.driver_profiles.items():
            dna = profile['dna_signature']
            if not dna.get('insufficient_data', False):
                features = [
                    dna.get('speed_vs_consistency_ratio', 0),
                    dna.get('track_adaptability', 0),
                    dna.get('consistency_index', 0),
                    dna.get('performance_variance', 0),
                    dna.get('speed_consistency', 0)
                ]
                dna_features.append(features)
                driver_ids.append(driver_id)
        
        if len(dna_features) < 3:
            print("❌ Insufficient data for comparative analysis")
            return
        
        # Perform clustering to identify driver archetypes
        dna_array = np.array(dna_features)
        dna_scaled = self.scaler.fit_transform(dna_array)
        
        # K-means clustering to identify driver types
        kmeans = KMeans(n_clusters=min(4, len(driver_ids)), random_state=42)
        clusters = kmeans.fit_predict(dna_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        dna_pca = pca.fit_transform(dna_scaled)
        
        # Create interactive scatter plot
        fig = px.scatter(
            x=dna_pca[:, 0], 
            y=dna_pca[:, 1],
            color=clusters,
            text=driver_ids,
            title='Driver DNA Clusters - Performance Archetypes',
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
        )
        
        fig.update_traces(textposition="top center")
        fig.update_layout(height=600)
        fig.show()
        
        # Print cluster analysis
        self._analyze_driver_clusters(clusters, driver_ids)
    
    def _analyze_driver_clusters(self, clusters, driver_ids):
        """Analyze and describe driver clusters"""
        print("\n🏁 DRIVER ARCHETYPE ANALYSIS")
        print("=" * 50)
        
        cluster_names = {
            0: "Speed Demons",
            1: "Consistency Masters", 
            2: "Track Specialists",
            3: "Balanced Racers"
        }
        
        for cluster_id in np.unique(clusters):
            cluster_drivers = [driver_ids[i] for i, c in enumerate(clusters) if c == cluster_id]
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            
            print(f"\n🏆 {cluster_name}")
            print(f"   Drivers: {', '.join(map(str, cluster_drivers))}")
            
            # Analyze cluster characteristics
            cluster_profiles = [self.driver_profiles[d] for d in cluster_drivers]
            self._describe_cluster_characteristics(cluster_profiles, cluster_name)
    
    def _describe_cluster_characteristics(self, profiles, cluster_name):
        """Describe the characteristics of a driver cluster"""
        if not profiles:
            return
            
        # Calculate average DNA characteristics for cluster
        dna_values = {
            'speed_vs_consistency_ratio': [],
            'track_adaptability': [],
            'consistency_index': [],
            'performance_variance': []
        }
        
        for profile in profiles:
            dna = profile['dna_signature']
            if not dna.get('insufficient_data', False):
                for key in dna_values.keys():
                    if key in dna:
                        dna_values[key].append(dna[key])
        
        print(f"   Characteristics:")
        for characteristic, values in dna_values.items():
            if values:
                avg_val = np.mean(values)
                print(f"   • {characteristic.replace('_', ' ').title()}: {avg_val:.2f}")

def main():
    """Main execution function"""
    print("🏁 MULTI-TRACK PERFORMANCE DNA ANALYZER")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PerformanceDNAAnalyzer()
    
    # Load and process data
    analyzer.load_track_data()
    analyzer.analyze_sector_performance()
    analyzer.create_driver_dna_profiles()
    
    # Create visualizations
    analyzer.visualize_driver_dna()
    
    print("\n🎯 Analysis complete! Driver DNA profiles generated.")
    print("💡 Each driver now has a unique performance fingerprint across all tracks.")

if __name__ == "__main__":
    main()