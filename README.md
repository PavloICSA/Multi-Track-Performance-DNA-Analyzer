# 🧬 Multi-Track Performance DNA Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-blue.svg)](https://www.microsoft.com/windows)

## 🎯 Project Overview

The **Multi-Track Performance DNA Analyzer** creates unique "performance fingerprints" for racing drivers across multiple tracks, identifying their strengths, weaknesses, and driving characteristics using advanced data analysis and machine learning techniques.

## 🚀 Quick Start

### For End Users
1. **Download**: Get the latest installer from [Releases](../../releases)
2. **Install**: Run `MTP_DNA_Analyzer_Setup.exe` and follow the wizard
3. **Launch**: Open from Start Menu or Desktop shortcut
4. **Analyze**: Click "Quick Start Guide" in the app for instructions

### For Developers
```bash
# Clone repository
git clone [YOUR_REPO_URL]
cd [REPO_NAME]

# Install dependencies
pip install -r requirements.txt

# Run application
python main_launcher.py
```

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for building the installer.

## 🎯 Key Features

### 1. **Driver DNA Profiling**
- **Speed vs Consistency Analysis**: Identifies whether drivers prioritize raw speed or consistent lap times
- **Track Adaptability Scoring**: Measures how well drivers adapt to different track layouts
- **Performance Variance Tracking**: Analyzes consistency across different racing conditions
- **Sector-Specific Strengths**: Identifies which parts of tracks drivers excel at

### 2. **Multi-Track Analysis**
- **Cross-Track Performance Comparison**: Compare driver performance across all 6 tracks
- **Track Difficulty Classification**: Automatically categorizes tracks as technical, high-speed, or balanced
- **Weather Impact Assessment**: Analyzes how weather conditions affect different drivers
- **Sector Time Breakdown**: Detailed analysis of S1, S2, S3 performance patterns

### 3. **Driver Archetypes**
The system identifies four main driver archetypes:

#### 🏎️ **Speed Demons**
- High raw speed but variable consistency
- Excel in qualifying but may struggle in race conditions
- **Training Focus**: Consistency and tire management

#### 🎯 **Consistency Masters** 
- Exceptional lap-to-lap consistency
- Strong race pace but may lack qualifying speed
- **Training Focus**: Peak performance and aggressive techniques

#### 🏁 **Track Specialists**
- Excel at specific track types but struggle with adaptability
- Strong at either technical or high-speed circuits
- **Training Focus**: Versatility and quick adaptation

#### ⚖️ **Balanced Racers**
- Well-rounded performance across all metrics
- Consistent performers across different track types
- **Training Focus**: Developing specialized strengths

### 4. **Interactive Visualizations**
- **DNA Radar Charts**: Visual representation of driver characteristics
- **Performance Heatmaps**: Track-by-track performance comparison
- **Cluster Analysis**: Groups drivers by similar performance patterns
- **Individual Driver Reports**: Detailed analysis for each driver

## 📊 Data Sources

The analyzer processes multiple data types:
- **Sector Times**: S1, S2, S3 performance data
- **Lap Times**: Complete lap time analysis with consistency metrics
- **Speed Data**: Top speeds and average speeds per sector
- **Weather Data**: Temperature, humidity, wind conditions
- **Race Results**: Final positions and race-specific performance

### 📁 Note on Large Telemetry Files

Due to GitHub's file size limitations (2GB per file), the large telemetry CSV files are not included in this repository. These files can be obtained from the official race data sources and manually added to the corresponding track folders:

**Required Telemetry Files:**
- `COTA/Race 1/R1_cota_telemetry_data.csv`
- `COTA/Race 2/R2_cota_telemetry_data.csv`
- `Road America/Race 1/R1_road_america_telemetry_data.csv`
- `Road America/Race 2/R2_road_america_telemetry_data.csv`
- `Sebring/Race 2/sebring_telemetry_R2.csv`
- `Sonoma/Race 1/sonoma_telemetry_R1.csv`
- `Sonoma/Race 2/sonoma_telemetry_R2.csv`
- `VIR/Race 1/R1_vir_telemetry_data.csv`
- `VIR/Race 2/R2_vir_telemetry_data.csv`
- `barber/R1_barber_telemetry_data.csv`
- `barber/R2_barber_telemetry_data.csv`

The application will function with the included lap time, sector time, and race results data. Full telemetry analysis features require these additional files to be placed in their respective directories.

## 🚀 Usage

### Quick Start
```bash
# Run the main DNA analyzer
python performance_dna_analyzer.py

# Generate interactive dashboard
python dna_dashboard.py

# Create detailed insights report
python dna_insights_generator.py
```

### Output Files
- `dna_dashboard.html` - Interactive multi-view dashboard
- `driver_[ID]_report.html` - Individual driver analysis
- `dna_insights_report.txt` - Comprehensive text report
- `dna_insights_data.json` - Machine-readable insights data

## 🧠 Key Insights Generated

### Track Analysis
- **Barber**: Technical circuit requiring precision and consistency
- **COTA**: Balanced circuit with mixed technical and speed sections
- **Road America**: High-speed circuit favoring aerodynamic efficiency
- **Sebring**: Technical endurance-style circuit
- **Sonoma**: Technical hillside circuit with elevation changes
- **VIR**: Mixed circuit with flowing technical sections

### Performance Patterns
1. **Sector Difficulty Rankings**: Identifies which sectors separate the field
2. **Speed vs Technical Preferences**: Classifies drivers by track type preference
3. **Consistency Patterns**: Identifies drivers who improve vs. decline over race distance
4. **Weather Adaptation**: Shows how different drivers handle varying conditions

### Training Recommendations
- **Personalized coaching programs** based on DNA archetype
- **Track-specific preparation strategies**
- **Weakness identification and improvement plans**
- **Strength amplification techniques**

## 🔬 Technical Implementation

### Machine Learning Components
- **K-Means Clustering**: Groups drivers into performance archetypes
- **Principal Component Analysis (PCA)**: Reduces dimensionality for visualization
- **Statistical Analysis**: Calculates performance metrics and correlations
- **Time Series Analysis**: Tracks performance evolution over time

### Data Processing Pipeline
1. **Data Loading**: Handles multiple CSV formats with different delimiters
2. **Time Conversion**: Converts MM:SS.mmm format to seconds
3. **Feature Engineering**: Creates derived metrics from raw data
4. **Normalization**: Standardizes metrics across different tracks
5. **DNA Signature Calculation**: Generates unique performance fingerprints

## 📈 Business Value

### For Drivers
- **Personalized training programs** based on individual DNA profiles
- **Track-specific preparation strategies**
- **Performance benchmarking** against similar driver archetypes
- **Weakness identification** with targeted improvement plans

### For Teams
- **Driver selection** based on track-specific requirements
- **Setup optimization** tailored to driver characteristics
- **Race strategy development** using DNA insights
- **Performance prediction** for upcoming races

### For Series Organizers
- **Competitive balance analysis**
- **Track difficulty assessment**
- **Driver development programs**
- **Fan engagement through driver personality insights**

## 🎯 Future Enhancements

1. **Real-Time DNA Tracking**: Live analysis during practice sessions
2. **Predictive Modeling**: Forecast race performance based on DNA profiles
3. **Telemetry Integration**: Incorporate detailed car data for deeper insights
4. **Machine Learning Evolution**: Continuously improve archetype classification
5. **Mobile Dashboard**: Real-time insights for trackside use

## 🏆 Results Summary

The analyzer successfully processed data from **6 tracks** and **38 drivers**, generating:
- **155 driver-track combinations** analyzed
- **4 distinct driver archetypes** identified
- **Comprehensive DNA profiles** for each driver
- **Interactive visualizations** for easy interpretation
- **Actionable training recommendations**

This system provides unprecedented insight into driver performance patterns, enabling data-driven decisions for training, strategy, and development programs.