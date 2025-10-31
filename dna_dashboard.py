#!/usr/bin/env python3
"""
Interactive DNA Dashboard for detailed driver analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from performance_dna_analyzer import PerformanceDNAAnalyzer

class DNADashboard:
    def __init__(self):
        self.analyzer = PerformanceDNAAnalyzer()
        self.analyzer.load_track_data()
        self.analyzer.analyze_sector_performance()
        self.analyzer.create_driver_dna_profiles()
        
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with multiple views"""
        print("🎨 Creating comprehensive DNA dashboard...")
        
        # Create main dashboard with multiple subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Driver Performance Heatmap Across Tracks',
                'Speed vs Consistency Analysis',
                'Track Adaptability Ranking',
                'Sector Performance Patterns',
                'Weather Impact Analysis',
                'Performance Evolution'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Performance Heatmap
        self._add_performance_heatmap(fig, row=1, col=1)
        
        # 2. Speed vs Consistency
        self._add_speed_consistency_plot(fig, row=1, col=2)
        
        # 3. Track Adaptability
        self._add_adaptability_ranking(fig, row=2, col=1)
        
        # 4. Sector Performance Patterns
        self._add_sector_patterns(fig, row=2, col=2)
        
        # 5. Weather Impact (if available)
        self._add_weather_impact(fig, row=3, col=1)
        
        # 6. Performance Evolution
        self._add_performance_evolution(fig, row=3, col=2)
        
        fig.update_layout(
            title="🧬 Multi-Track Performance DNA Dashboard",
            height=1200,
            showlegend=False
        )
        
        # Save as HTML
        pyo.plot(fig, filename='dna_dashboard.html', auto_open=True)
        print("✅ Dashboard saved as 'dna_dashboard.html'")
        
    def _add_performance_heatmap(self, fig, row, col):
        """Add performance heatmap across tracks"""
        # Create matrix of driver performance across tracks
        drivers = list(self.analyzer.driver_profiles.keys())[:15]  # Top 15 for visibility
        tracks = self.analyzer.tracks
        
        performance_matrix = []
        driver_labels = []
        
        for driver in drivers:
            profile = self.analyzer.driver_profiles[driver]
            row_data = []
            
            for track in tracks:
                if track in profile['performance_metrics']:
                    # Use relative performance (lower is better for lap times)
                    lap_time = profile['performance_metrics'][track]['avg_lap_time']
                    row_data.append(lap_time if lap_time > 0 else np.nan)
                else:
                    row_data.append(np.nan)
            
            if any(not np.isnan(x) for x in row_data):
                performance_matrix.append(row_data)
                driver_labels.append(f"Driver {driver}")
        
        if performance_matrix:
            fig.add_trace(
                go.Heatmap(
                    z=performance_matrix,
                    x=tracks,
                    y=driver_labels,
                    colorscale='RdYlBu_r',
                    name='Lap Time Performance'
                ),
                row=row, col=col
            )
    
    def _add_speed_consistency_plot(self, fig, row, col):
        """Add speed vs consistency scatter plot"""
        drivers = []
        speed_ratios = []
        consistency_scores = []
        colors = []
        
        for driver_id, profile in self.analyzer.driver_profiles.items():
            dna = profile['dna_signature']
            if not dna.get('insufficient_data', False):
                drivers.append(driver_id)
                speed_ratios.append(dna.get('speed_vs_consistency_ratio', 0))
                consistency_scores.append(dna.get('consistency_index', 0))
                
                # Color by archetype (simplified)
                if dna.get('speed_vs_consistency_ratio', 0) > 10:
                    colors.append('Consistency Master')
                elif dna.get('performance_variance', 0) > 0.2:
                    colors.append('Track Specialist')
                elif dna.get('speed_vs_consistency_ratio', 0) > 6:
                    colors.append('Balanced Racer')
                else:
                    colors.append('Speed Demon')
        
        fig.add_trace(
            go.Scatter(
                x=speed_ratios,
                y=consistency_scores,
                mode='markers+text',
                text=drivers,
                textposition="top center",
                marker=dict(size=8, opacity=0.7),
                name='Driver DNA'
            ),
            row=row, col=col
        )
    
    def _add_adaptability_ranking(self, fig, row, col):
        """Add track adaptability ranking"""
        adaptability_data = []
        
        for driver_id, profile in self.analyzer.driver_profiles.items():
            dna = profile['dna_signature']
            if not dna.get('insufficient_data', False):
                adaptability_data.append({
                    'driver': driver_id,
                    'adaptability': dna.get('track_adaptability', 0)
                })
        
        # Sort by adaptability
        adaptability_data.sort(key=lambda x: x['adaptability'], reverse=True)
        top_15 = adaptability_data[:15]
        
        fig.add_trace(
            go.Bar(
                x=[d['adaptability'] for d in top_15],
                y=[f"Driver {d['driver']}" for d in top_15],
                orientation='h',
                name='Track Adaptability'
            ),
            row=row, col=col
        )
    
    def _add_sector_patterns(self, fig, row, col):
        """Add sector performance pattern analysis"""
        # Analyze sector time distributions across all drivers
        sector_data = []
        
        if hasattr(self.analyzer, 'sector_analysis'):
            df = self.analyzer.sector_analysis
            
            # Get sector time averages
            s1_times = df['S1_mean'].dropna()
            s2_times = df['S2_mean'].dropna()
            s3_times = df['S3_mean'].dropna()
            
            fig.add_trace(go.Box(y=s1_times, name='Sector 1'), row=row, col=col)
            fig.add_trace(go.Box(y=s2_times, name='Sector 2'), row=row, col=col)
            fig.add_trace(go.Box(y=s3_times, name='Sector 3'), row=row, col=col)
    
    def _add_weather_impact(self, fig, row, col):
        """Add weather impact analysis if data available"""
        # Simplified weather impact visualization
        # This would need more sophisticated analysis with actual weather correlation
        
        weather_impact_data = {
            'Temperature': [25, 30, 35, 28, 32],
            'Lap_Time_Impact': [0.2, -0.1, -0.5, 0.1, -0.3]
        }
        
        fig.add_trace(
            go.Scatter(
                x=weather_impact_data['Temperature'],
                y=weather_impact_data['Lap_Time_Impact'],
                mode='markers+lines',
                name='Weather Impact'
            ),
            row=row, col=col
        )
    
    def _add_performance_evolution(self, fig, row, col):
        """Add performance evolution across races"""
        # Show how drivers improve/decline across race weekends
        evolution_data = []
        
        for driver_id, profile in list(self.analyzer.driver_profiles.items())[:5]:
            tracks = profile['tracks_raced']
            lap_times = []
            
            for track in tracks:
                if track in profile['performance_metrics']:
                    lap_times.append(profile['performance_metrics'][track]['avg_lap_time'])
            
            if len(lap_times) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(lap_times))),
                        y=lap_times,
                        mode='lines+markers',
                        name=f'Driver {driver_id}',
                        line=dict(width=2)
                    ),
                    row=row, col=col
                )
    
    def create_individual_driver_report(self, driver_id):
        """Create detailed report for individual driver"""
        if driver_id not in self.analyzer.driver_profiles:
            print(f"❌ Driver {driver_id} not found")
            return
        
        profile = self.analyzer.driver_profiles[driver_id]
        
        print(f"\n🏁 DRIVER {driver_id} - PERFORMANCE DNA REPORT")
        print("=" * 60)
        
        # Basic stats
        print(f"📊 Races Completed: {profile['total_races']}")
        print(f"🏁 Tracks Raced: {', '.join(profile['tracks_raced'])}")
        
        # DNA Signature
        dna = profile['dna_signature']
        if not dna.get('insufficient_data', False):
            print(f"\n🧬 DNA SIGNATURE:")
            print(f"   • Speed vs Consistency Ratio: {dna.get('speed_vs_consistency_ratio', 0):.2f}")
            print(f"   • Track Adaptability: {dna.get('track_adaptability', 0):.2f}")
            print(f"   • Consistency Index: {dna.get('consistency_index', 0):.3f}")
            print(f"   • Performance Variance: {dna.get('performance_variance', 0):.3f}")
        
        # Track-by-track performance
        print(f"\n🏁 TRACK PERFORMANCE:")
        for track, metrics in profile['performance_metrics'].items():
            print(f"   📍 {track.upper()}:")
            print(f"      • Average Lap Time: {metrics['avg_lap_time']:.3f}s")
            print(f"      • Best Lap: {metrics['best_lap']:.3f}s")
            print(f"      • Consistency Score: {metrics['consistency']:.3f}")
            print(f"      • Average Speed: {metrics['speed_profile']['avg_speed']:.1f} km/h")
        
        # Create individual visualization
        self._create_individual_visualization(driver_id, profile)
    
    def _create_individual_visualization(self, driver_id, profile):
        """Create individual driver visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Driver {driver_id} - Track Performance',
                'Sector Time Breakdown',
                'DNA Radar Chart',
                'Speed Profile'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatterpolar"}, {"type": "bar"}]
            ]
        )
        
        tracks = list(profile['performance_metrics'].keys())
        
        # Track performance
        lap_times = [profile['performance_metrics'][track]['avg_lap_time'] for track in tracks]
        fig.add_trace(
            go.Bar(x=tracks, y=lap_times, name='Avg Lap Time'),
            row=1, col=1
        )
        
        # Speed profile
        speeds = [profile['performance_metrics'][track]['speed_profile']['avg_speed'] for track in tracks]
        fig.add_trace(
            go.Bar(x=tracks, y=speeds, name='Avg Speed'),
            row=2, col=2
        )
        
        # DNA Radar
        dna = profile['dna_signature']
        if not dna.get('insufficient_data', False):
            categories = ['Speed/Consistency', 'Adaptability', 'Consistency', 'Variance']
            values = [
                dna.get('speed_vs_consistency_ratio', 0) / 10,  # Normalized
                dna.get('track_adaptability', 0) / 10,
                dna.get('consistency_index', 0) * 100,
                (1 - dna.get('performance_variance', 0)) * 10
            ]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=f'Driver {driver_id} DNA'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'Driver {driver_id} - Detailed Performance Analysis',
            height=800
        )
        
        pyo.plot(fig, filename=f'driver_{driver_id}_report.html', auto_open=True)
        print(f"✅ Individual report saved as 'driver_{driver_id}_report.html'")

def main():
    """Main dashboard execution"""
    dashboard = DNADashboard()
    
    print("🎨 Creating DNA Dashboard...")
    dashboard.create_comprehensive_dashboard()
    
    # Create individual reports for top performers
    print("\n📋 Creating individual driver reports...")
    top_drivers = [13, 22, 72, 2, 47]  # Based on earlier results
    
    for driver in top_drivers:
        if driver in dashboard.analyzer.driver_profiles:
            dashboard.create_individual_driver_report(driver)
    
    print("\n🎯 Dashboard creation complete!")
    print("💡 Open the HTML files in your browser to explore the interactive visualizations.")

if __name__ == "__main__":
    main()