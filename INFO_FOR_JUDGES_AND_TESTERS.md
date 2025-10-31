# MTP DNA Analyzer - Hackathon Submission Guide

## 🎯 Project Overview
**Multi-Track Performance DNA Analyzer** - An AI-powered racing performance analysis tool that creates unique "performance fingerprints" for drivers across multiple tracks, identifying strengths, weaknesses, and driving characteristics.

## 📦 Installation & Testing

### For Judges and Testers

#### Quick Start (Recommended)
1. **Download the installer**: [MTP_DNA_Analyzer_Setup.exe](LINK_TO_RELEASE)
2. **Run the installer** and follow the setup wizard
3. **Launch** the application from Start Menu or Desktop shortcut
4. **Click "Quick Start Guide"** in the application for usage instructions

#### System Requirements
- Windows 10/11 (64-bit)
- ~20 GB free disk space
- No additional software required (all dependencies bundled)

### For Developers

#### Building from Source
```cmd
# 1. Clone the repository
git clone https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer.git
cd Multi-Track-Performance-DNA-Analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python main_launcher.py

# 4. Build installer (optional)
# Install Inno Setup from: https://jrsoftware.org/isdl.php
build_installer.bat
```

## 🎬 Demo Video
**[Link to 3-minute demo video on YouTube]**

The video demonstrates:
- Application launch;
- Built-in dataset analysis workflow;
- Custom dataset import;
- Driver DNA profiling and insights;
- Performance visualizations and reports.

## 📊 Datasets Used

### Built-in Dataset
- **Source**: Multi-track racing telemetry data.
- **Tracks**: 6 circuits (Barber, COTA, Road America, Sebring, Sonoma, VIR).
- **Drivers**: 38 drivers.
- **Data Points**: 155+ driver-track combinations.
- **Metrics**: Sector times, lap times, speed data, consistency metrics.

### Custom Dataset Support
- Users can import their own CSV racing data;
- Flexible format support with automatic detection;
- Validation and preprocessing included;
- .zip format support for multiple files.

## 🔑 Key Features

### 1. Driver DNA Profiling
- **Speed vs Consistency Analysis**: Identifies driver priorities
- **Track Adaptability Scoring**: Measures adaptation to different layouts
- **Performance Variance Tracking**: Analyzes consistency patterns
- **Sector-Specific Strengths**: Identifies track section expertise

### 2. AI-Powered Insights
- **K-Means Clustering**: Groups drivers into _4 archetypes_:
  - Speed Demons - Fastest drivers;
  - Consistency Masters - Most consistent drivers;
  - Track Specialists - Drivers who excel in specific tracks;
  - Balanced Racers - Drivers who are balanced across tracks.
- **PCA Analysis**: Dimensionality reduction for visualization.
- **Statistical Analysis**: Performance metrics and correlations.

### 3. Interactive Visualizations
- DNA Radar Charts;
- Performance Heatmaps;
- Cluster Analysis Plots;
- Individual Driver Reports;
- Track Comparison Dashboards.

### 4. Professional GUI
- Modern dark-themed interface (CustomTkinter);
- Intuitive navigation and workflow;
- Real-time progress tracking;
- Export capabilities (text reports, visualizations).

## 🏗️ Technical Architecture

### Technology Stack
- **Language**: Python 3.13
- **GUI Framework**: CustomTkinter, Tkinter
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Packaging**: PyInstaller, Inno Setup

### Project Structure
```
MTP-DNA-Analyzer/
├── main_launcher.py              # Application entry point
├── dna_analyzer_gui.py           # Main GUI application
├── performance_dna_analyzer.py   # Core analysis engine
├── dna_model_trainer.py          # ML model training
├── dna_dashboard.py              # Dashboard generation
├── dna_insights_generator.py     # Insights engine
├── config.py                     # Configuration
├── requirements.txt              # Python dependencies
├── build_installer.bat           # Build automation
├── build_spec.spec               # PyInstaller config
├── installer_script.iss          # Inno Setup config
├── icon.ico                      # Application icon
├── README.md                     # Documentation
├── BUILD_INSTRUCTIONS.md         # Build guide
└── [data folders]/               # Track data
```

## 🎓 Innovation & Impact

### Problem Solved
Racing teams and drivers lack accessible tools to:
- Identify individual performance patterns across multiple tracks;
- Understand driver-specific strengths and weaknesses;
- Make data-driven training and strategy decisions;
- Benchmark performance against similar driver profiles.

### Solution Approach
- **Automated DNA Profiling**: Converts raw telemetry into actionable insights.
- **Cross-Track Analysis**: Identifies patterns across different circuit types.
- **Personalized Recommendations**: Tailored training programs based on archetype.
- **Accessible Interface**: Professional tool accessible to non-technical users.

### Business Value
- **For Drivers**: Personalized training programs and performance benchmarking.
- **For Teams**: Data-driven driver selection and setup optimization.
- **For Series Organizers**: Competitive balance analysis and driver development.
- **For Fans**: Enhanced engagement through driver personality insights.

## 🔬 Machine Learning Components

### Models Implemented
1. **K-Means Clustering**: Driver archetype classification.
2. **PCA**: Feature reduction and visualization.
3. **Statistical Analysis**: Performance metric calculation.
4. **Time Series Analysis**: Performance evolution tracking.

### Training Process
- Automated feature engineering from raw telemetry;
- Normalization across different track characteristics;
- Cross-validation for model robustness;
- Continuous learning capability for new data.

## 📈 Results & Validation

### Analysis Capabilities
- ✅ 155 driver-track combinations analyzed
- ✅ 4 distinct driver archetypes identified
- ✅ Comprehensive DNA profiles for 38 drivers
- ✅ Interactive visualizations for easy interpretation
- ✅ Actionable training recommendations

### Performance Metrics
- Fast analysis: ~30 seconds for full dataset
- Accurate clustering: High silhouette scores
- Intuitive visualizations: User-tested interface
- Scalable: Handles custom datasets of varying sizes

## 🚀 Future Enhancements

1. **Predictive Modeling**: Robust ML-driven race performance forecasting.
2. **Mobile Application**: Developing an Android application for smartphones.
3. **Cross-platform Support**: Ensuring compatibility across Windows, macOS, and Linux.

## 📝 License & Attribution

### Open Source Components
- Python (PSF License)
- CustomTkinter (MIT License)
- Pandas, NumPy (BSD License)
- Scikit-learn (BSD License)
- Matplotlib, Seaborn (BSD License)
- PyTorch (BSD License)

### Original Work
All application code, analysis algorithms, and GUI design are original work created specifically for this hackathon.

## 👥 Team Information
- **Team/Developer:** _Pavlo Lykhovyd_
- **Location:** _UKRAINE_
- **Bio:** _Self-taught software developer, with a strong passion for AI/ML and programming._
- **Contact:** _pavlolykhovyd55@gmail.com_

## 📧 Code, Demo, Installation Files
- **Repository**: [[GitHub URL]](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/)
- **Demo Video**: [YouTube URL]
- **Installer Download**: [Release URL](https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases/download/v1.0.0/MTP_DNA_Analyzer_Setup.exe)

## 🏆 Hackathon Category
**Driver Training & Insights**

### Why This Category?
This application directly addresses the **Driver Training & Insights** category by:
- ✅ _Identifying areas for improvement_: DNA-style profiling reveals specific weaknesses in speed, consistency, and track adaptability;
- ✅ _Optimizing racing performance_: Sector-specific analysis shows exactly where drivers excel or struggle;
- ✅ _Understanding performance patterns_: Cross-track analysis reveals driver characteristics and tendencies;
- ✅ _Providing actionable training plans_: Personalized recommendations based on driver archetype (Speed Demon, Consistency Master, Track Specialist, or Balanced Racer, based on the major driver features, extracted from racing data);
- ✅ _Benchmarking against peers_: Cluster analysis groups similar drivers for comparative insights.

---

**Note for Judges**: The application is fully functional and ready for testing. No special hardware or proprietary software required. All dependencies are bundled in the installer. For any issues or questions, please contact the developer.

