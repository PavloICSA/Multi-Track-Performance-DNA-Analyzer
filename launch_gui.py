#!/usr/bin/env python3
"""
GUI Launcher for Multi-Track Performance DNA Analyzer
Simple launcher with dependency checking
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        'tkinter',
        'customtkinter', 
        'PIL',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'torch',
        'sklearn'
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages"""
    
    if not packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(packages)}")
    
    # Package name mapping
    package_mapping = {
        'PIL': 'pillow',
        'sklearn': 'scikit-learn'
    }
    
    for package in packages:
        install_name = package_mapping.get(package, package)
        
        if package == 'tkinter':
            print(f"   ⚠️  tkinter should be included with Python. Please check your Python installation.")
            continue
            
        try:
            print(f"   📦 Installing {install_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', install_name])
            print(f"   ✅ {install_name} installed successfully")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {install_name}")
            return False
    
    return True

def check_data_files():
    """Check if data files are available"""
    
    print("\n📊 Checking data files...")
    
    tracks = ['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']
    available_tracks = 0
    
    for track in tracks:
        track_path = Path(track)
        if track_path.exists():
            csv_files = list(track_path.glob('**/*.csv')) + list(track_path.glob('**/*.CSV'))
            if csv_files:
                available_tracks += 1
                print(f"   ✅ {track} ({len(csv_files)} files)")
            else:
                print(f"   ⚠️  {track} - Directory exists but no CSV files found")
        else:
            print(f"   ❌ {track} - Directory not found")
    
    print(f"\n📈 Data Status: {available_tracks}/6 tracks available")
    
    if available_tracks < 5:
        print("   ⚠️  Warning: Need at least 5 tracks for full analysis")
        return False
    
    return True

def launch_gui():
    """Launch the main GUI application"""
    
    print("\n🚀 Launching DNA Analyzer GUI...")
    
    try:
        from dna_analyzer_gui import DNAAnalyzerGUI
        
        print("   ✅ GUI modules loaded successfully")
        print("   🎨 Starting graphical interface...")
        
        app = DNAAnalyzerGUI()
        app.run()
        
    except ImportError as e:
        print(f"   ❌ Failed to import GUI modules: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Failed to launch GUI: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    
    print("🧬 MULTI-TRACK PERFORMANCE DNA ANALYZER")
    print("=" * 50)
    print("GUI Launcher v1.0")
    print()
    
    # Step 1: Check dependencies
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies detected: {', '.join(missing_deps)}")
        
        response = input("\n📦 Would you like to install missing packages automatically? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            if not install_missing_packages(missing_deps):
                print("\n❌ Failed to install some packages. Please install manually:")
                for package in missing_deps:
                    install_name = 'pillow' if package == 'PIL' else 'scikit-learn' if package == 'sklearn' else package
                    print(f"   pip install {install_name}")
                return
        else:
            print("\n❌ Cannot proceed without required dependencies.")
            return
    
    # Step 2: Check data files
    data_available = check_data_files()
    
    if not data_available:
        response = input("\n⚠️  Limited data available. Continue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("\n❌ Exiting. Please ensure data files are in the correct directories.")
            return
    
    # Step 3: Launch GUI
    print("\n" + "=" * 50)
    success = launch_gui()
    
    if success:
        print("\n✅ GUI session completed successfully")
    else:
        print("\n❌ GUI launch failed")

if __name__ == "__main__":
    main()