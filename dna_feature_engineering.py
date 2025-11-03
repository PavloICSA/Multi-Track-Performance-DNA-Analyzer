#!/usr/bin/env python3
"""
DNA Feature Engineering Module
Provides consistent feature extraction pipeline for training and inference
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DNAFeatureEngineering:
    """
    Feature engineering pipeline for driver performance DNA analysis.
    Extracts and transforms raw racing data into model-ready features.
    """
    
    def __init__(self):
        """Initialize feature engineering pipeline"""
        self.track_types = {
            'barber': 'technical',
            'COTA': 'mixed',
            'Road America': 'high_speed',
            'Sebring': 'technical',
            'Sonoma': 'technical',
            'VIR': 'mixed'
        }
    
    def load_and_process_csv(self, file_path: Union[str, Path], track_name: str = None) -> pd.DataFrame:
        """
        Load CSV with automatic delimiter detection and column cleaning
        
        Args:
            file_path: Path to CSV file
            track_name: Optional track name to add to dataframe
            
        Returns:
            Processed DataFrame with cleaned columns
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try different delimiters
        delimiters = [';', ',', '\t', '|']
        df = None
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin-1')
                # Valid data should have more than 5 columns
                if len(df.columns) > 5:
                    break
            except Exception:
                continue
        
        if df is None or len(df.columns) <= 5:
            raise ValueError(f"Could not parse CSV file with any delimiter: {file_path}")
        
        # Clean column names - remove leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Add track name if provided
        if track_name:
            df['track'] = track_name
        
        return df
    
    def convert_time_to_seconds(self, time_value: Any) -> float:
        """
        Convert time format (MM:SS.mmm or seconds) to seconds
        
        Args:
            time_value: Time value in various formats (string, float, int)
            
        Returns:
            Time in seconds as float, or NaN if conversion fails
        """
        if pd.isna(time_value) or time_value == '':
            return np.nan
        
        try:
            # If already a number, return it
            if isinstance(time_value, (int, float)):
                return float(time_value)
            
            # If string with colon, parse MM:SS.mmm format
            if isinstance(time_value, str) and ':' in time_value:
                parts = time_value.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            
            # Try direct conversion
            return float(time_value)
            
        except (ValueError, IndexError, AttributeError):
            return np.nan
    
    def validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame contains required columns
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, list_of_missing_columns)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        is_valid = len(missing_columns) == 0
        return is_valid, missing_columns

    def extract_driver_features(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract per-driver features from sector data
        
        Args:
            sector_data: DataFrame with columns NUMBER, LAP_TIME, S1, S2, S3, KPH, track
            
        Returns:
            DataFrame with aggregated driver features per track:
            - driver_id
            - track
            - avg_lap_time, std_lap_time, min_lap_time, lap_count
            - avg_s1, std_s1, min_s1
            - avg_s2, std_s2, min_s2
            - avg_s3, std_s3, min_s3
            - avg_speed, max_speed
        """
        if sector_data.empty:
            return pd.DataFrame()
        
        # Validate required columns
        required_cols = ['NUMBER', 'LAP_TIME', 'S1', 'S2', 'S3', 'KPH', 'track']
        is_valid, missing = self.validate_columns(sector_data, required_cols)
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert time columns to seconds
        time_cols = ['LAP_TIME', 'S1', 'S2', 'S3']
        df = sector_data.copy()
        
        for col in time_cols:
            df[col] = df[col].apply(self.convert_time_to_seconds)
        
        # Convert KPH to numeric
        df['KPH'] = pd.to_numeric(df['KPH'], errors='coerce')
        
        # Handle missing values and outliers
        # Remove rows where lap time is missing or unrealistic (< 30 seconds or > 300 seconds)
        df = df[df['LAP_TIME'].notna()]
        df = df[(df['LAP_TIME'] >= 30) & (df['LAP_TIME'] <= 300)]
        
        # Remove rows where speed is missing or unrealistic (< 50 km/h or > 400 km/h)
        df = df[df['KPH'].notna()]
        df = df[(df['KPH'] >= 50) & (df['KPH'] <= 400)]
        
        # Calculate statistical features per driver-track combination
        driver_features = df.groupby(['NUMBER', 'track']).agg({
            'LAP_TIME': ['mean', 'std', 'min', 'count'],
            'S1': ['mean', 'std', 'min'],
            'S2': ['mean', 'std', 'min'],
            'S3': ['mean', 'std', 'min'],
            'KPH': ['mean', 'max']
        }).round(3)
        
        # Flatten column names
        driver_features.columns = ['_'.join(col).strip() for col in driver_features.columns]
        driver_features = driver_features.reset_index()
        
        # Rename columns to match expected format
        driver_features = driver_features.rename(columns={
            'NUMBER': 'driver_id',
            'LAP_TIME_mean': 'avg_lap_time',
            'LAP_TIME_std': 'std_lap_time',
            'LAP_TIME_min': 'min_lap_time',
            'LAP_TIME_count': 'lap_count',
            'S1_mean': 'avg_s1',
            'S1_std': 'std_s1',
            'S1_min': 'min_s1',
            'S2_mean': 'avg_s2',
            'S2_std': 'std_s2',
            'S2_min': 'min_s2',
            'S3_mean': 'avg_s3',
            'S3_std': 'std_s3',
            'S3_min': 'min_s3',
            'KPH_mean': 'avg_speed',
            'KPH_max': 'max_speed'
        })
        
        # Fill NaN std values with 0 (happens when only 1 lap)
        std_cols = ['std_lap_time', 'std_s1', 'std_s2', 'std_s3']
        for col in std_cols:
            if col in driver_features.columns:
                driver_features[col] = driver_features[col].fillna(0)
        
        return driver_features

    def calculate_dna_features(self, driver_track_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DNA signature features for each driver
        
        Args:
            driver_track_data: DataFrame with driver features per track
            
        Returns:
            DataFrame with DNA signature values:
            - driver_id
            - speed_vs_consistency_ratio
            - track_adaptability
            - consistency_index
            - performance_variance
            - speed_consistency
            - technical_track_performance
            - high_speed_track_performance
            - mixed_track_performance
            - sector_balance_score
        """
        if driver_track_data.empty:
            return pd.DataFrame()
        
        # Group by driver to calculate cross-track DNA features
        drivers = driver_track_data['driver_id'].unique()
        dna_records = []
        
        for driver_id in drivers:
            driver_data = driver_track_data[driver_track_data['driver_id'] == driver_id]
            
            # Need at least 2 tracks for meaningful DNA calculation
            if len(driver_data) < 2:
                continue
            
            # Extract performance metrics across tracks
            lap_times = driver_data['avg_lap_time'].values
            lap_time_stds = driver_data['std_lap_time'].values
            speeds = driver_data['avg_speed'].values
            tracks = driver_data['track'].values
            
            # Calculate DNA signature components
            dna = {
                'driver_id': driver_id
            }
            
            # 1. Speed vs Consistency Ratio
            # Higher speed with lower lap time variance = higher ratio
            mean_speed = np.mean(speeds)
            lap_time_variance = np.std(lap_times)
            dna['speed_vs_consistency_ratio'] = mean_speed / (lap_time_variance + 0.001)
            
            # 2. Track Adaptability
            # Lower variance in relative performance across tracks = higher adaptability
            # Normalized lap time variance (coefficient of variation)
            cv_lap_time = lap_time_variance / (np.mean(lap_times) + 0.001)
            dna['track_adaptability'] = 1.0 / (cv_lap_time + 0.001)
            
            # 3. Consistency Index
            # Average of consistency scores (inverse of std) across tracks
            consistency_scores = 1.0 / (lap_time_stds + 0.001)
            dna['consistency_index'] = np.mean(consistency_scores)
            
            # 4. Performance Variance
            # Coefficient of variation of lap times across tracks
            dna['performance_variance'] = cv_lap_time
            
            # 5. Speed Consistency
            # Coefficient of variation of speeds across tracks
            speed_variance = np.std(speeds)
            dna['speed_consistency'] = speed_variance / (mean_speed + 0.001)
            
            # 6-8. Track Specialization Scores
            track_specialization = self._calculate_track_type_performance(driver_data)
            dna.update(track_specialization)
            
            # 9. Sector Balance Score
            sector_balance = self._calculate_sector_balance(driver_data)
            dna['sector_balance_score'] = sector_balance
            
            dna_records.append(dna)
        
        dna_df = pd.DataFrame(dna_records)
        return dna_df
    
    def _calculate_track_type_performance(self, driver_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance scores for different track types
        
        Args:
            driver_data: DataFrame with driver performance per track
            
        Returns:
            Dictionary with technical_track_performance, high_speed_track_performance, 
            mixed_track_performance scores
        """
        performance_by_type = {
            'technical': [],
            'high_speed': [],
            'mixed': []
        }
        
        for _, row in driver_data.iterrows():
            track = row['track']
            track_type = self.track_types.get(track, 'mixed')
            
            # Use inverse of lap time as performance score (lower time = better)
            # Normalize by dividing by mean to get relative performance
            lap_time = row['avg_lap_time']
            if lap_time > 0:
                performance_by_type[track_type].append(lap_time)
        
        # Calculate average performance for each track type
        # Lower average lap time = better performance
        # Convert to a score where higher is better
        result = {}
        all_lap_times = []
        for times in performance_by_type.values():
            all_lap_times.extend(times)
        
        overall_mean = np.mean(all_lap_times) if all_lap_times else 1.0
        
        for track_type in ['technical', 'high_speed', 'mixed']:
            times = performance_by_type[track_type]
            if times:
                # Relative performance: overall_mean / type_mean
                # If driver is faster on this type, score > 1.0
                type_mean = np.mean(times)
                result[f'{track_type}_track_performance'] = overall_mean / (type_mean + 0.001)
            else:
                result[f'{track_type}_track_performance'] = 1.0  # Neutral score
        
        return result
    
    def _calculate_sector_balance(self, driver_data: pd.DataFrame) -> float:
        """
        Calculate sector balance score (how evenly distributed sector times are)
        
        Args:
            driver_data: DataFrame with driver performance per track
            
        Returns:
            Sector balance score (lower variance = more balanced)
        """
        sector_times = []
        
        for _, row in driver_data.iterrows():
            s1 = row.get('avg_s1', 0)
            s2 = row.get('avg_s2', 0)
            s3 = row.get('avg_s3', 0)
            
            if s1 > 0 and s2 > 0 and s3 > 0:
                # Calculate relative sector times (as proportion of total)
                total = s1 + s2 + s3
                sector_proportions = [s1/total, s2/total, s3/total]
                # Variance of proportions (lower = more balanced)
                sector_times.append(np.std(sector_proportions))
        
        if sector_times:
            # Return inverse of average variance (higher = more balanced)
            return 1.0 / (np.mean(sector_times) + 0.001)
        else:
            return 1.0  # Neutral score

    def create_archetype_labels(self, dna_features: pd.DataFrame) -> pd.Series:
        """
        Create archetype labels based on DNA features
        
        Uses rule-based classification to assign drivers to one of four archetypes:
        - Speed Demon: High speed, lower consistency
        - Consistency Master: High consistency, moderate speed
        - Track Specialist: Strong specialization in specific track types
        - Balanced Racer: Well-rounded across all metrics
        
        Args:
            dna_features: DataFrame with DNA signature features
            
        Returns:
            Series with archetype labels for each driver
        """
        if dna_features.empty:
            return pd.Series(dtype=str)
        
        archetypes = []
        
        for _, row in dna_features.iterrows():
            archetype = self._classify_driver_archetype(row)
            archetypes.append(archetype)
        
        return pd.Series(archetypes, index=dna_features.index)
    
    def _classify_driver_archetype(self, dna_row: pd.Series) -> str:
        """
        Classify a single driver into an archetype based on DNA features
        
        Args:
            dna_row: Series with DNA features for one driver
            
        Returns:
            Archetype label string
        """
        # Extract key metrics
        speed_consistency_ratio = dna_row.get('speed_vs_consistency_ratio', 0)
        track_adaptability = dna_row.get('track_adaptability', 0)
        consistency_index = dna_row.get('consistency_index', 0)
        performance_variance = dna_row.get('performance_variance', 0)
        speed_consistency = dna_row.get('speed_consistency', 0)
        
        # Track specialization scores
        technical_perf = dna_row.get('technical_track_performance', 1.0)
        high_speed_perf = dna_row.get('high_speed_track_performance', 1.0)
        mixed_perf = dna_row.get('mixed_track_performance', 1.0)
        
        # Calculate specialization variance (high variance = specialist)
        track_perfs = [technical_perf, high_speed_perf, mixed_perf]
        specialization_variance = np.std(track_perfs)
        max_specialization = max(track_perfs)
        
        # Decision tree for archetype classification
        # Using relative thresholds based on typical racing data patterns
        
        # Track Specialist: Significant variance in track type performance
        # Strong in one type, weaker in others
        if specialization_variance > 0.08 and max_specialization > 1.05:
            return 'Track Specialist'
        
        # Speed Demon: High speed/consistency ratio, accepts more variance
        # Prioritizes raw speed over lap-to-lap consistency
        # High speed consistency (variance in speed) indicates aggressive driving
        if speed_consistency_ratio > 10.0 and speed_consistency > 0.05:
            return 'Speed Demon'
        
        # Consistency Master: High consistency index, low performance variance
        # Very consistent lap times, low variance across tracks
        if consistency_index > 0.08 and performance_variance < 0.015:
            return 'Consistency Master'
        
        # Alternative Consistency Master: Very low performance variance
        if performance_variance < 0.01:
            return 'Consistency Master'
        
        # Balanced Racer: Moderate in all metrics, good track adaptability
        # Default category for well-rounded drivers
        return 'Balanced Racer'
    
    def process_full_pipeline(self, sector_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Run the complete feature engineering pipeline
        
        Args:
            sector_data: Raw sector data with lap times, sectors, speeds
            
        Returns:
            Tuple of (driver_features, dna_features, archetype_labels)
        """
        # Step 1: Extract driver features
        driver_features = self.extract_driver_features(sector_data)
        
        # Step 2: Calculate DNA signatures
        dna_features = self.calculate_dna_features(driver_features)
        
        # Step 3: Create archetype labels
        archetype_labels = self.create_archetype_labels(dna_features)
        
        return driver_features, dna_features, archetype_labels


def main():
    """Example usage of DNAFeatureEngineering"""
    # DNA Feature Engineering Module
    
    # Initialize feature engineering
    feature_eng = DNAFeatureEngineering()
    
    # Example: Load and process a single track
    try:
        # Load barber track data
        df = feature_eng.load_and_process_csv(
            'barber/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV',
            track_name='barber'
        )
        # Loaded rows from barber track
        
        # Extract driver features
        driver_features = feature_eng.extract_driver_features(df)
        # Extracted features for driver-track combinations
        
        # Calculate DNA signatures
        dna_features = feature_eng.calculate_dna_features(driver_features)
        # Calculated DNA signatures for drivers
        
        # Create archetype labels
        archetypes = feature_eng.create_archetype_labels(dna_features)
        # Assigned archetypes to drivers
        
        # Display sample results
        # Sample DNA Features
        # Archetype Distribution
        
    except Exception as e:
        # Error occurred
        pass


if __name__ == "__main__":
    main()
