# MTP DNA Analyzer - Build Instructions

## Overview
This guide will help you create a production-ready `setup.exe` installer for the MTP DNA Analyzer application.

## Prerequisites

### 1. Python Environment
- Python 3.13 installed
- All dependencies from `requirements.txt` installed
- Run: `pip install -r requirements.txt`

### 2. PyInstaller
- Will be automatically installed by the build script
- Or manually install: `pip install pyinstaller`

### 3. Inno Setup (for creating setup.exe)
- Download from: https://jrsoftware.org/isdl.php
- Install Inno Setup 6 (free)
- Default installation path: `C:\Program Files (x86)\Inno Setup 6\`

## Build Process

### Quick Build (Automated)
Simply run the build script:
```cmd
build_installer.bat
```

This will:
1. Check and install PyInstaller if needed
2. Clean previous builds
3. Build the executable using PyInstaller
4. Create the installer using Inno Setup
5. Output: `installer_output\MTP_DNA_Analyzer_Setup.exe`

### Manual Build (Step by Step)

#### Step 1: Install PyInstaller
```cmd
pip install pyinstaller
```

#### Step 2: Build the Executable
```cmd
pyinstaller build_spec.spec --clean --noconfirm
```

This creates:
- `dist\MTP DNA Analyzer\` - Folder with all application files
- `dist\MTP DNA Analyzer\MTP DNA Analyzer.exe` - Main executable

#### Step 3: Test the Executable
```cmd
cd "dist\MTP DNA Analyzer"
"MTP DNA Analyzer.exe"
```

#### Step 4: Create the Installer
```cmd
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer_script.iss
```

This creates:
- `installer_output\MTP_DNA_Analyzer_Setup.exe` - Final installer

## What Gets Packaged

### Application Files
- All Python scripts (dna_*.py, etc.)
- Configuration files (config.py)
- Icon file (icon.ico)
- README documentation

### Data Folders
- barber/
- COTA/
- Road America/
- Sebring/
- Sonoma/
- VIR/
- models/
- training_artifacts/

### Dependencies
All Python packages are bundled:
- customtkinter (GUI framework)
- pandas, numpy (data processing)
- matplotlib, seaborn, plotly (visualization)
- torch, sklearn (machine learning)
- And all other requirements

## Installer Features

### Installation Wizard
- Professional Windows installer interface
- User can choose installation directory
- Default: `C:\Program Files\MTP DNA Analyzer\`

### Desktop Integration
- Start Menu shortcut created automatically
- Optional desktop shortcut (user choice)
- Custom icon from `icon.ico`

### Post-Installation
- Option to launch application immediately
- All files properly organized
- Uninstaller automatically created

## File Structure After Installation

```
C:\Program Files\MTP DNA Analyzer\
├── MTP DNA Analyzer.exe          (Main launcher)
├── icon.ico                       (Application icon)
├── README.md                      (Documentation)
├── _internal\                     (Python runtime & dependencies)
├── barber\                        (Track data)
├── COTA\
├── Road America\
├── Sebring\
├── Sonoma\
├── VIR\
├── models\                        (ML models)
└── training_artifacts\
```

## Testing the Installer

### Before Distribution
1. Build the installer using `build_installer.bat`
2. Test on a clean Windows machine (VM recommended)
3. Run `MTP_DNA_Analyzer_Setup.exe`
4. Follow installation wizard
5. Launch application from Start Menu or Desktop
6. Verify all features work correctly
7. Test uninstallation

### Common Issues

#### PyInstaller Build Fails
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Try: `pip install --upgrade pyinstaller`

#### Missing DLL Errors
- Usually auto-resolved by PyInstaller
- If persistent, add to `hiddenimports` in `build_spec.spec`

#### Icon Not Showing
- Verify `icon.ico` exists in root directory
- Check icon format (must be .ico, not .png)

#### Inno Setup Not Found
- Install from: https://jrsoftware.org/isdl.php
- Update path in `build_installer.bat` if installed elsewhere

## Customization

### Change Application Name
Edit `installer_script.iss`:
```iss
#define MyAppName "Your App Name"
```

### Change Version
Edit `installer_script.iss`:
```iss
#define MyAppVersion "2.0.0"
```

### Add/Remove Files
Edit `build_spec.spec` in the `datas` section:
```python
datas=[
    ('your_file.txt', '.'),
    ('your_folder', 'your_folder'),
]
```

### Change Icon
Replace `icon.ico` with your custom icon file

## Distribution

### Final Deliverable
- File: `installer_output\MTP_DNA_Analyzer_Setup.exe`
- Size: ~200-500 MB (includes all dependencies)
- Portable: Single file, no external dependencies needed

### User Requirements
- Windows 10/11 (64-bit)
- ~1 GB free disk space
- Administrator rights for installation

### Installation Steps for End Users
1. Download `MTP_DNA_Analyzer_Setup.exe`
2. Double-click to run
3. Follow installation wizard
4. Choose installation directory
5. Optionally create desktop shortcut
6. Click Install
7. Launch application

## Support

### Build Issues
- Check Python version: `python --version`
- Verify dependencies: `pip list`
- Review build logs in `build\` folder

### Runtime Issues
- Check logs in application folder
- Verify data files are present
- Ensure sufficient disk space

## Version History

### v1.0.0 (Current)
- Full GUI application
- Multi-track analysis
- ML model integration
- Professional installer




