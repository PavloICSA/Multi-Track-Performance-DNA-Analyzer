# 🧬 Multi-Track Performance DNA Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-blue.svg)](https://www.microsoft.com/windows)
[![🏁 Release](https://img.shields.io/github/v/release/PavloICSA/Multi-Track-Performance-DNA-Analyzer?label=🏁%20Release&color=red)](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases)
[![🏎️ Tracks](https://img.shields.io/badge/🏎️%20Tracks-6-orange)](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer)
[![👥 Drivers](https://img.shields.io/badge/👥%20Drivers-38-green)](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer)

> An AI-powered racing performance analysis tool that creates unique "performance fingerprints" for drivers across multiple tracks, identifying strengths, weaknesses, and driving characteristics.

## 🎯 Project Overview

The **Multi-Track Performance DNA Analyzer** revolutionizes driver performance analysis by creating comprehensive "DNA profiles" that reveal individual racing characteristics. Using advanced machine learning and statistical analysis, it processes multi-track racing data to identify driver archetypes, track-specific strengths, and personalized training opportunities.

**Key Innovation**: Unlike traditional lap time analysis, this tool creates holistic performance fingerprints that capture speed, consistency, adaptability, and sector-specific expertise across different track types.

## 📦 Installation

### For End Users (Recommended)

**Download the latest release**: [MTP_DNA_Analyzer_Setup.exe](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases/latest)

#### System Requirements
- Windows 10/11 (64-bit)
- ~20 GB free disk space
- No additional software required (all dependencies bundled)

#### Installation Steps
1. Download `MTP_DNA_Analyzer_Setup.exe` from [Releases](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases)
2. Run the installer and follow the setup wizard
3. Launch from Start Menu or Desktop shortcut
4. Click "Quick Start Guide" in the application for usage instructions

### For Developers

```bash
# Clone repository
git clone https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer.git
cd Multi-Track-Performance-DNA-Analyzer

# Install dependencies
pip install -r requirements.txt

# Run application
python main_launcher.py
```

**Building the installer**: See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for detailed build instructions.

**Creating releases**: See [HOW_TO_CREATE_RELEASE.md](HOW_TO_CREATE_RELEASE.md) for release management.

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

### GUI Application (Recommended)
Launch the application and follow the intuitive workflow:
1. **Load Data**: Use built-in dataset or import custom CSV files
2. **Analyze**: Click "Analyze Performance DNA" to process data
3. **Explore**: View DNA profiles, archetypes, and visualizations
4. **Export**: Generate reports and save insights

### Command Line (For Developers)
```bash
# Run the main DNA analyzer
python performance_dna_analyzer.py

# Generate interactive dashboard
python dna_dashboard.py

# Create detailed insights report
python dna_insights_generator.py

# Train/retrain ML models
python train_dna_model.py
```

### Output Files
- `dna_dashboard.html` - Interactive multi-view dashboard
- `driver_[ID]_report.html` - Individual driver analysis
- `dna_insights_report.txt` - Comprehensive text report
- `dna_insights_data.json` - Machine-readable insights data
- `models/` - Trained ML models for archetype classification

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

### Technology Stack
- **Language**: Python 3.13
- **GUI Framework**: CustomTkinter (modern dark-themed interface)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Packaging**: PyInstaller, Inno Setup

### Machine Learning Components
- **K-Means Clustering**: Groups drivers into 4 performance archetypes
- **Principal Component Analysis (PCA)**: Reduces dimensionality for visualization
- **Statistical Analysis**: Calculates performance metrics and correlations
- **Time Series Analysis**: Tracks performance evolution over time
- **Feature Engineering**: Automated extraction of 20+ performance indicators

### Data Processing Pipeline
1. **Data Loading**: Handles multiple CSV formats with different delimiters
2. **Time Conversion**: Converts MM:SS.mmm format to seconds
3. **Feature Engineering**: Creates derived metrics from raw data
4. **Normalization**: Standardizes metrics across different tracks
5. **DNA Signature Calculation**: Generates unique performance fingerprints
6. **Model Inference**: Classifies drivers into archetypes
7. **Insight Generation**: Produces actionable recommendations

### Project Structure
```
Multi-Track-Performance-DNA-Analyzer/
├── main_launcher.py              # Application entry point
├── dna_analyzer_gui.py           # Main GUI application
├── performance_dna_analyzer.py   # Core analysis engine
├── dna_model_trainer.py          # ML model training
├── dna_model_inference.py        # Model prediction
├── dna_dashboard.py              # Dashboard generation
├── dna_insights_generator.py     # Insights engine
├── dna_feature_engineering.py    # Feature extraction
├── dna_explainability.py         # Model interpretability
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── build_installer.bat           # Build automation
├── create_release.bat            # Release automation
├── README.md                     # This file
├── BUILD_INSTRUCTIONS.md         # Build guide
├── HOW_TO_CREATE_RELEASE.md      # Release guide
├── INFO_FOR_JUDGES_AND_TESTERS.md # Hackathon submission
└── [track folders]/              # Race data by track
```

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
2. **Predictive Modeling**: Robust ML-driven race performance forecasting
3. **Cross-Platform Support**: macOS and Linux compatibility
4. **Mobile Application**: Android/iOS app for trackside use
5. **Enhanced Telemetry Integration**: Deeper car data analysis
6. **Cloud Sync**: Multi-device data synchronization
7. **Team Collaboration**: Shared insights and reports

## 🏆 Results Summary

The analyzer successfully processed data from **6 tracks** and **38 drivers**, generating:
- ✅ **155 driver-track combinations** analyzed
- ✅ **4 distinct driver archetypes** identified (Speed Demons, Consistency Masters, Track Specialists, Balanced Racers)
- ✅ **Comprehensive DNA profiles** for each driver
- ✅ **Interactive visualizations** for easy interpretation
- ✅ **Actionable training recommendations** based on individual profiles
- ✅ **Professional GUI** with modern dark-themed interface
- ✅ **Fast analysis**: ~30 seconds for full dataset processing

This system provides unprecedented insight into driver performance patterns, enabling data-driven decisions for training, strategy, and development programs.

## 📚 Documentation

- **[README.md](README.md)** - This file (overview and quick start)
- **[INFO_FOR_JUDGES_AND_TESTERS.md](INFO_FOR_JUDGES_AND_TESTERS.md)** - Hackathon submission guide
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - How to build the installer
- **[HOW_TO_CREATE_RELEASE.md](HOW_TO_CREATE_RELEASE.md)** - Release management guide
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** - Version history and changes

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/issues)
- **Email**: pavlolykhovyd55@gmail.com
- **Developer**: Pavlo Lykhovyd
- **Location**: Ukraine

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

This project uses open-source components:
- Python (PSF License)
- CustomTkinter (MIT License)
- Pandas, NumPy, Scikit-learn (BSD License)
- Matplotlib, Seaborn (BSD License)
- PyTorch (BSD License)

## 🙏 Acknowledgments

Built for the **Driver Training & Insights** category, this application addresses the need for:
- ✅ Identifying areas for improvement through DNA profiling
- ✅ Optimizing racing performance with sector-specific analysis
- ✅ Understanding performance patterns across multiple tracks
- ✅ Providing actionable training plans based on driver archetypes
- ✅ Benchmarking against peers through cluster analysis

---

**Made with ❤️ by Pavlo Lykhovyd | Ukraine 🇺🇦**