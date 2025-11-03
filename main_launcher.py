#!/usr/bin/env python3
"""
Main Launcher for MTP DNA Analyzer
GUI-only launcher without console dependencies
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def check_critical_dependencies():
    """Check only critical dependencies needed to start GUI"""
    
    critical_packages = {
        'tkinter': 'tkinter',
        'customtkinter': 'customtkinter',
        'PIL': 'Pillow',
    }
    
    missing = []
    
    for module_name, package_name in critical_packages.items():
        try:
            if module_name == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    
    return missing

def show_error_dialog(title, message):
    """Show error dialog using tkinter"""
    try:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except:
        # Fallback to console if tkinter fails
        # ERROR occurred
        pass

def launch_application():
    """Launch the main DNA Analyzer GUI"""
    
    try:
        # Import the main GUI module
        from dna_analyzer_gui import DNAAnalyzerGUI
        
        # Create and run the application
        app = DNAAnalyzerGUI()
        app.run()
        
        return True
        
    except ImportError as e:
        error_msg = (
            f"Failed to import required modules:\n\n{str(e)}\n\n"
            "Please ensure all application files are present."
        )
        show_error_dialog("Import Error", error_msg)
        return False
        
    except Exception as e:
        error_msg = (
            f"Failed to launch application:\n\n{str(e)}\n\n"
            f"Details:\n{traceback.format_exc()}"
        )
        show_error_dialog("Launch Error", error_msg)
        return False

def main():
    """Main entry point"""
    
    # Set working directory to application directory
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        application_path = os.path.dirname(sys.executable)
        
        # PyInstaller puts data files in _internal subfolder
        internal_path = os.path.join(application_path, '_internal')
        if os.path.exists(internal_path):
            # Change to _internal where data folders are located
            os.chdir(internal_path)
        else:
            os.chdir(application_path)
    else:
        # Running as script
        application_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(application_path)
    
    # Check critical dependencies
    missing_deps = check_critical_dependencies()
    
    if missing_deps:
        error_msg = (
            "Missing critical dependencies:\n\n" +
            "\n".join(f"  • {pkg}" for pkg in missing_deps) +
            "\n\nPlease reinstall the application."
        )
        show_error_dialog("Missing Dependencies", error_msg)
        sys.exit(1)
    
    # Launch the application
    success = launch_application()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
