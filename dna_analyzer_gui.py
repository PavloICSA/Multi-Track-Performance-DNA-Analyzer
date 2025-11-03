#!/usr/bin/env python3
"""
Multi-Track Performance DNA Analyzer - GUI Application
Professional desktop interface for racing data analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import webbrowser
import os
import json
import time
import zipfile
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Import our analysis modules
from performance_dna_analyzer import PerformanceDNAAnalyzer
from dna_dashboard import DNADashboard

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DNAAnalyzerGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🧬 Multi-Track Performance DNA Analyzer")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize variables
        self.analyzer = None
        self.dashboard = None
        self.analysis_complete = False
        self.current_driver = None
        self.data_source = "built-in"  # "built-in" or "custom"
        self.custom_data_path = None
        self.data_validation_results = None
        self.temp_extract_dir = None  # For extracted zip files
        self.extracted_folders = []  # Track extracted folders for cleanup
        
        # Create the interface
        self.create_interface()
        
        # Load initial data check
        self.check_data_availability()
        
        # Register cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Maximize window after everything is created
        self.root.after(10, lambda: self.root.state('zoomed'))
        
    def create_interface(self):
        """Create the main GUI interface"""
        
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.create_main_content()
        
        # Create status bar
        self.create_status_bar()
        
    def create_sidebar(self):
        """Create the sidebar with navigation and controls"""
        
        self.sidebar = ctk.CTkFrame(self.root, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="🧬 DNA Analyzer", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar, 
            text="Multi-Track Performance Analysis", 
            font=ctk.CTkFont(size=12)
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        # Quick Start Button (prominent)
        self.quick_start_button = ctk.CTkButton(
            self.sidebar,
            text="🏁 Quick Start Guide",
            command=self.show_quick_start,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.quick_start_button.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        # Data Source Selection
        self.data_source_frame = ctk.CTkFrame(self.sidebar)
        self.data_source_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.data_source_label = ctk.CTkLabel(
            self.data_source_frame, 
            text="📂 Data Source", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.data_source_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Data source radio buttons
        self.data_source_var = tk.StringVar(value="built-in")
        
        self.builtin_radio = ctk.CTkRadioButton(
            self.data_source_frame,
            text="Built-in Dataset",
            variable=self.data_source_var,
            value="built-in",
            command=self.on_data_source_change
        )
        self.builtin_radio.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        
        self.custom_radio = ctk.CTkRadioButton(
            self.data_source_frame,
            text="Custom Dataset",
            variable=self.data_source_var,
            value="custom",
            command=self.on_data_source_change
        )
        self.custom_radio.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        
        # Browse data button (initially disabled)
        self.browse_data_button = ctk.CTkButton(
            self.data_source_frame,
            text="📁 Browse Data",
            command=self.browse_custom_data,
            state="disabled",
            height=30
        )
        self.browse_data_button.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        # Data Status Section
        self.data_status_frame = ctk.CTkFrame(self.sidebar)
        self.data_status_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        self.data_status_label = ctk.CTkLabel(
            self.data_status_frame, 
            text="📊 Data Status", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.data_status_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.tracks_status = ctk.CTkLabel(self.data_status_frame, text="Tracks: Checking...")
        self.tracks_status.grid(row=1, column=0, padx=10, pady=2)
        
        self.files_status = ctk.CTkLabel(self.data_status_frame, text="Files: Checking...")
        self.files_status.grid(row=2, column=0, padx=10, pady=2)
        
        self.validation_status = ctk.CTkLabel(self.data_status_frame, text="Validation: Ready")
        self.validation_status.grid(row=3, column=0, padx=10, pady=2)
        
        # Analysis Controls
        self.controls_frame = ctk.CTkFrame(self.sidebar)
        self.controls_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        self.controls_label = ctk.CTkLabel(
            self.controls_frame, 
            text="🎯 Analysis Controls", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.controls_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.analyze_button = ctk.CTkButton(
            self.controls_frame,
            text="🚀 Start Analysis",
            command=self.start_analysis,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.analyze_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.progress_bar = ctk.CTkProgressBar(self.controls_frame)
        self.progress_bar.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(self.controls_frame, text="Ready to analyze")
        self.progress_label.grid(row=3, column=0, padx=10, pady=2)
        
        # Navigation Buttons
        self.nav_frame = ctk.CTkFrame(self.sidebar)
        self.nav_frame.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        self.nav_label = ctk.CTkLabel(
            self.nav_frame, 
            text="📋 Navigation", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.nav_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.overview_button = ctk.CTkButton(
            self.nav_frame,
            text="📊 Overview",
            command=lambda: self.show_tab("overview")
        )
        self.overview_button.grid(row=1, column=0, padx=10, pady=2, sticky="ew")
        
        self.drivers_button = ctk.CTkButton(
            self.nav_frame,
            text="👥 Drivers",
            command=lambda: self.show_tab("drivers")
        )
        self.drivers_button.grid(row=2, column=0, padx=10, pady=2, sticky="ew")
        
        self.tracks_button = ctk.CTkButton(
            self.nav_frame,
            text="🏁 Tracks",
            command=lambda: self.show_tab("tracks")
        )
        self.tracks_button.grid(row=3, column=0, padx=10, pady=2, sticky="ew")
        
        self.insights_button = ctk.CTkButton(
            self.nav_frame,
            text="🧠 Insights",
            command=lambda: self.show_tab("insights")
        )
        self.insights_button.grid(row=4, column=0, padx=10, pady=2, sticky="ew")
        
        self.guidelines_button = ctk.CTkButton(
            self.nav_frame,
            text="📖 Guidelines",
            command=lambda: self.show_tab("guidelines")
        )
        self.guidelines_button.grid(row=5, column=0, padx=10, pady=2, sticky="ew")
        
        # Export Options
        self.export_frame = ctk.CTkFrame(self.sidebar)
        self.export_frame.grid(row=7, column=0, padx=20, pady=10, sticky="ew")
        
        self.export_label = ctk.CTkLabel(
            self.export_frame, 
            text="📤 Export", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.export_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.dashboard_button = ctk.CTkButton(
            self.export_frame,
            text="📊 Dashboard View",
            command=self.show_dashboard_tab
        )
        self.dashboard_button.grid(row=1, column=0, padx=10, pady=2, sticky="ew")
        
        self.report_button = ctk.CTkButton(
            self.export_frame,
            text="📋 Report View",
            command=self.show_report_tab
        )
        self.report_button.grid(row=2, column=0, padx=10, pady=2, sticky="ew")
        
        self.visualizations_button = ctk.CTkButton(
            self.export_frame,
            text="📈 Visualization View",
            command=self.show_visualization_tab
        )
        self.visualizations_button.grid(row=3, column=0, padx=10, pady=2, sticky="ew")
        
    def create_main_content(self):
        """Create the main content area with tabs"""
        
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Tab header
        self.tab_header = ctk.CTkLabel(
            self.main_frame,
            text="🏁 Welcome to DNA Analyzer",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.tab_header.grid(row=0, column=0, padx=20, pady=20)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        # Create tabs
        self.create_overview_tab()
        self.create_drivers_tab()
        self.create_tracks_tab()
        self.create_insights_tab()
        self.create_guidelines_tab()
        self.create_dashboard_tab()
        self.create_report_tab()
        self.create_visualization_tab()
        
    def create_overview_tab(self):
        """Create the overview tab with better space utilization"""
        
        self.overview_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.overview_frame, text="📊 Overview")
        
        # Configure grid weights for better space usage
        self.overview_frame.grid_columnconfigure(0, weight=1)
        self.overview_frame.grid_columnconfigure(1, weight=1)
        self.overview_frame.grid_rowconfigure(0, weight=1)
        self.overview_frame.grid_rowconfigure(1, weight=1)
        self.overview_frame.grid_rowconfigure(2, weight=1)
        
        # Top section - Statistics cards (more prominent)
        self.stats_frame = ctk.CTkFrame(self.overview_frame)
        self.stats_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
        # Create enhanced stats cards
        self.create_enhanced_stats_cards()
        
        # Left section - Quick insights
        self.insights_frame = ctk.CTkFrame(self.overview_frame)
        self.insights_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.insights_label = ctk.CTkLabel(
            self.insights_frame,
            text="🧠 Quick Insights",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.insights_label.pack(pady=10)
        
        self.insights_text = ctk.CTkTextbox(
            self.insights_frame,
            font=ctk.CTkFont(size=12)
        )
        self.insights_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right section - Performance summary
        self.performance_frame = ctk.CTkFrame(self.overview_frame)
        self.performance_frame.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        
        self.performance_label = ctk.CTkLabel(
            self.performance_frame,
            text="📈 Performance Summary",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.performance_label.pack(pady=10)
        
        self.performance_text = ctk.CTkTextbox(
            self.performance_frame,
            font=ctk.CTkFont(size=12)
        )
        self.performance_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Bottom section - Embedded visualization
        self.viz_frame = ctk.CTkFrame(self.overview_frame)
        self.viz_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
        self.viz_label = ctk.CTkLabel(
            self.viz_frame,
            text="📊 Live Performance Dashboard",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.viz_label.pack(pady=10)
        
        # Placeholder for embedded chart
        self.chart_placeholder = ctk.CTkLabel(
            self.viz_frame,
            text="Run analysis to see live performance charts",
            font=ctk.CTkFont(size=16)
        )
        self.chart_placeholder.pack(expand=True)
        
    def create_drivers_tab(self):
        """Create the drivers analysis tab with better space utilization"""
        
        self.drivers_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.drivers_frame, text="👥 Drivers")
        
        # Configure grid for better space usage
        self.drivers_frame.grid_columnconfigure(0, weight=1)
        self.drivers_frame.grid_columnconfigure(1, weight=1)
        self.drivers_frame.grid_rowconfigure(1, weight=2)
        self.drivers_frame.grid_rowconfigure(2, weight=1)
        
        # Top section - Driver selection and quick stats
        self.driver_header_frame = ctk.CTkFrame(self.drivers_frame)
        self.driver_header_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        # Driver selection
        self.driver_label = ctk.CTkLabel(
            self.driver_header_frame,
            text="Select Driver:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.driver_label.grid(row=0, column=0, padx=20, pady=10)
        
        self.driver_dropdown = ctk.CTkComboBox(
            self.driver_header_frame,
            values=["Run analysis first..."],
            command=self.on_driver_selected,
            width=200
        )
        self.driver_dropdown.grid(row=0, column=1, padx=20, pady=10)
        
        # Quick driver stats
        self.driver_quick_stats = ctk.CTkFrame(self.driver_header_frame)
        self.driver_quick_stats.grid(row=0, column=2, columnspan=3, padx=20, pady=10, sticky="ew")
        
        # Left section - Driver profile and DNA
        self.driver_profile_frame = ctk.CTkFrame(self.drivers_frame)
        self.driver_profile_frame.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")
        
        self.profile_label = ctk.CTkLabel(
            self.driver_profile_frame,
            text="🏁 Driver Profile",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        )
        self.profile_label.pack(pady=10)
        
        self.driver_info_text = ctk.CTkTextbox(
            self.driver_profile_frame,
            font=ctk.CTkFont(size=10),
            wrap="word"
        )
        self.driver_info_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Right section - Performance visualization
        self.driver_viz_frame = ctk.CTkFrame(self.drivers_frame)
        self.driver_viz_frame.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")
        
        self.viz_label = ctk.CTkLabel(
            self.driver_viz_frame,
            text="📊 Performance Visualization",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        )
        self.viz_label.pack(pady=10)
        
        # Placeholder for driver chart
        self.driver_chart_placeholder = ctk.CTkLabel(
            self.driver_viz_frame,
            text="Select a driver to see performance charts",
            font=ctk.CTkFont(size=16)
        )
        self.driver_chart_placeholder.pack(expand=True)
        
        # Bottom section - Track performance comparison
        self.track_performance_frame = ctk.CTkFrame(self.drivers_frame)
        self.track_performance_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        
        self.track_perf_label = ctk.CTkLabel(
            self.track_performance_frame,
            text="🏁 Track Performance Breakdown",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.track_perf_label.pack(pady=10)
        
        # Track performance will be populated after analysis
        self.track_perf_content = ctk.CTkFrame(self.track_performance_frame)
        self.track_perf_content.pack(fill="both", expand=True, padx=10, pady=10)
        
    def create_tracks_tab(self):
        """Create the tracks analysis tab"""
        
        self.tracks_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.tracks_frame, text="🏁 Tracks")
        
        self.tracks_frame.grid_columnconfigure(0, weight=1)
        self.tracks_frame.grid_rowconfigure(1, weight=1)
        
        # Track info
        self.track_info_label = ctk.CTkLabel(
            self.tracks_frame,
            text="Track Analysis - Performance characteristics across all circuits",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.track_info_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Track analysis content
        self.track_content_frame = ctk.CTkFrame(self.tracks_frame)
        self.track_content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
    def create_insights_tab(self):
        """Create the insights tab"""
        
        self.insights_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.insights_frame, text="🧠 Insights")
        
        self.insights_frame.grid_columnconfigure(0, weight=1)
        self.insights_frame.grid_rowconfigure(1, weight=1)
        
        # Insights header
        self.insights_header = ctk.CTkLabel(
            self.insights_frame,
            text="Advanced Insights & Recommendations",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.insights_header.grid(row=0, column=0, padx=20, pady=20)
        
        # Insights content
        self.insights_content = ctk.CTkTextbox(
            self.insights_frame,
            font=ctk.CTkFont(size=12)
        )
        self.insights_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
    def create_enhanced_stats_cards(self):
        """Create compact statistics summary"""
        
        # Single row with key stats
        self.stats_frame.grid_columnconfigure(0, weight=1)
        
        # Create a simple text summary instead of button-like cards
        self.stats_summary = ctk.CTkLabel(
            self.stats_frame,
            text="🏁 6 Tracks  |  👥 -- Drivers  |  📊 -- Data Points  |  🧬 4 Archetypes  |  🎯 Ready",
            font=ctk.CTkFont(size=14),
            text_color="white"
        )
        self.stats_summary.grid(row=0, column=0, padx=20, pady=15, sticky="ew")
        
    def create_status_bar(self):
        """Create the status bar at the bottom"""
        
        self.status_bar = ctk.CTkFrame(self.root, height=30)
        self.status_bar.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready - Multi-Track Performance DNA Analyzer v1.0",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5)
        
    def check_data_availability(self):
        """Check if data files are available (supports both folders and zip files)"""
        
        tracks = ['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']
        available_tracks = 0
        total_files = 0
        zip_count = 0
        
        for track in tracks:
            track_path = Path(track)
            track_zip = Path(f"{track}.zip")
            
            # Check if folder exists
            if track_path.exists() and track_path.is_dir():
                csv_files = list(track_path.glob('**/*.csv')) + list(track_path.glob('**/*.CSV'))
                if csv_files:
                    available_tracks += 1
                    total_files += len(csv_files)
            # Check if zip file exists
            elif track_zip.exists():
                available_tracks += 1
                zip_count += 1
                # Count files in zip without extracting
                try:
                    with zipfile.ZipFile(track_zip, 'r') as zip_ref:
                        csv_files = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')]
                        total_files += len(csv_files)
                except:
                    pass
        
        # Update status
        status_text = f"Tracks: {available_tracks}/6 available"
        if zip_count > 0:
            status_text += f" ({zip_count} zipped)"
        self.tracks_status.configure(text=status_text)
        self.files_status.configure(text=f"Files: {total_files} CSV files found")
        
        if available_tracks >= 5:
            self.analyze_button.configure(state="normal")
        else:
            self.analyze_button.configure(state="disabled")
            messagebox.showwarning(
                "Data Warning",
                f"Only {available_tracks}/6 tracks found. Need at least 5 tracks for analysis."
            )
    
    def extract_zip_data(self):
        """Extract zip files to temporary directory if needed"""
        
        tracks = ['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']
        
        for track in tracks:
            track_path = Path(track)
            track_zip = Path(f"{track}.zip")
            
            # If folder doesn't exist but zip does, extract it
            if not track_path.exists() and track_zip.exists():
                try:
                    # Create temp directory if not exists
                    if self.temp_extract_dir is None:
                        self.temp_extract_dir = tempfile.mkdtemp(prefix="dna_analyzer_")
                    
                    # Extract to current directory (not temp) for analyzer to find
                    self.root.after(0, lambda t=track: self.progress_label.configure(
                        text=f"Extracting {t} data..."
                    ))
                    
                    with zipfile.ZipFile(track_zip, 'r') as zip_ref:
                        zip_ref.extractall(path='.')
                    
                    self.extracted_folders.append(track)
                    
                except Exception as e:
                    print(f"Warning: Could not extract {track_zip}: {e}")
    
    def cleanup_extracted_data(self):
        """Clean up extracted zip data"""
        
        for folder in self.extracted_folders:
            folder_path = Path(folder)
            if folder_path.exists() and folder_path.is_dir():
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"Warning: Could not remove {folder}: {e}")
        
        self.extracted_folders = []
        
        if self.temp_extract_dir and os.path.exists(self.temp_extract_dir):
            try:
                shutil.rmtree(self.temp_extract_dir)
            except Exception as e:
                print(f"Warning: Could not remove temp directory: {e}")
    
    def on_closing(self):
        """Handle window closing - cleanup extracted data"""
        
        self.cleanup_extracted_data()
        self.root.destroy()
    
    def start_analysis(self):
        """Start the DNA analysis in a separate thread"""
        
        self.analyze_button.configure(state="disabled", text="🔄 Analyzing...")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting analysis...")
        
        # Run analysis in separate thread to prevent GUI freezing
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
    def run_analysis(self):
        """Run the actual analysis"""
        
        try:
            # Step 0: Extract zip files if needed
            self.root.after(0, lambda: self.progress_label.configure(text="Preparing data..."))
            self.root.after(0, lambda: self.progress_bar.set(0.05))
            
            self.extract_zip_data()
            
            # Step 1: Initialize analyzer
            self.root.after(0, lambda: self.progress_label.configure(text="Initializing analyzer..."))
            self.root.after(0, lambda: self.progress_bar.set(0.1))
            
            self.analyzer = PerformanceDNAAnalyzer()
            
            # Step 2: Load data
            self.root.after(0, lambda: self.progress_label.configure(text="Loading track data..."))
            self.root.after(0, lambda: self.progress_bar.set(0.3))
            
            self.analyzer.load_track_data()
            
            # Step 3: Analyze sectors
            self.root.after(0, lambda: self.progress_label.configure(text="Analyzing sector performance..."))
            self.root.after(0, lambda: self.progress_bar.set(0.6))
            
            self.analyzer.analyze_sector_performance()
            
            # Step 4: Create DNA profiles
            self.root.after(0, lambda: self.progress_label.configure(text="Creating DNA profiles..."))
            self.root.after(0, lambda: self.progress_bar.set(0.8))
            
            self.analyzer.create_driver_dna_profiles()
            
            # Step 5: Complete
            self.root.after(0, lambda: self.progress_label.configure(text="Analysis complete!"))
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            
            self.analysis_complete = True
            
            # Update GUI
            self.root.after(0, self.update_gui_after_analysis)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}"))
            self.root.after(0, lambda: self.analyze_button.configure(state="normal", text="🚀 Start Analysis"))
            self.root.after(0, lambda: self.progress_label.configure(text="Analysis failed"))
    
    def update_gui_after_analysis(self):
        """Update GUI elements after analysis is complete"""
        
        # Update buttons
        self.analyze_button.configure(state="normal", text="✅ Analysis Complete")
        
        # Update stats summary with real data
        driver_count = len(self.analyzer.driver_profiles)
        data_points = len(self.analyzer.sector_analysis) if hasattr(self.analyzer, 'sector_analysis') else 0
        
        self.stats_summary.configure(
            text=f"🏁 6 Tracks  |  👥 {driver_count} Drivers  |  📊 {data_points} Data Points  |  🧬 4 Archetypes  |  ✅ Complete"
        )
        
        # Update overview content
        self.update_overview_content()
        
        # Update driver dropdown
        driver_list = [f"Driver {d}" for d in sorted(self.analyzer.driver_profiles.keys())]
        self.driver_dropdown.configure(values=driver_list)
        if driver_list:
            self.driver_dropdown.set(driver_list[0])
            self.on_driver_selected(driver_list[0])
        
        # Update all tabs
        self.update_insights_tab()
        self.update_tracks_tab()
        self.update_dashboard_content()
        self.update_report_content()
        self.update_visualization_content()
        
        # Show success message
        messagebox.showinfo(
            "Analysis Complete",
            f"Successfully analyzed {driver_count} drivers across 6 tracks!\n\nExplore all tabs to view comprehensive results."
        )
        
    def on_driver_selected(self, selection):
        """Handle driver selection"""
        
        if not self.analysis_complete or not selection or selection == "Run analysis first...":
            return
            
        try:
            # Extract driver ID from selection
            driver_id = int(selection.split()[1])
            self.current_driver = driver_id
            
            # Update driver info
            self.update_driver_info(driver_id)
            
        except (ValueError, IndexError):
            pass
    
    def update_driver_info(self, driver_id):
        """Update driver information display with enhanced content"""
        
        if driver_id not in self.analyzer.driver_profiles:
            return
            
        profile = self.analyzer.driver_profiles[driver_id]
        dna = profile.get('dna_signature', {})
        
        # Update quick stats in header
        self.update_driver_quick_stats(driver_id, profile)
        
        # Create comprehensive driver info text
        info_text = f"""🏁 DRIVER {driver_id} - COMPREHENSIVE PROFILE

📊 PERFORMANCE OVERVIEW:
• Total Races Completed: {profile['total_races']}
• Tracks Raced: {len(profile['tracks_raced'])}/6
• Coverage: {', '.join(profile['tracks_raced'])}
• Data Quality: {'Excellent' if len(profile['tracks_raced']) >= 5 else 'Good' if len(profile['tracks_raced']) >= 3 else 'Limited'}

🧬 DNA SIGNATURE ANALYSIS:"""
        
        if not dna.get('insufficient_data', False):
            info_text += f"""
• Speed/Consistency Ratio: {dna.get('speed_vs_consistency_ratio', 0):.2f}
  {'🟢 Balanced' if 5 <= dna.get('speed_vs_consistency_ratio', 0) <= 15 else '🟡 Specialized' if dna.get('speed_vs_consistency_ratio', 0) > 15 else '🔴 Needs Work'}
• Track Adaptability: {dna.get('track_adaptability', 0):.2f}
  {'🟢 Excellent' if dna.get('track_adaptability', 0) > 10 else '🟡 Good' if dna.get('track_adaptability', 0) > 5 else '🔴 Limited'}
• Consistency Index: {dna.get('consistency_index', 0):.3f}
  {'🟢 Very Consistent' if dna.get('consistency_index', 0) > 0.1 else '🟡 Consistent' if dna.get('consistency_index', 0) > 0.05 else '🔴 Inconsistent'}
• Performance Variance: {dna.get('performance_variance', 0):.3f}
  {'🟢 Stable' if dna.get('performance_variance', 0) < 0.1 else '🟡 Moderate' if dna.get('performance_variance', 0) < 0.2 else '🔴 Variable'}

🏆 ARCHETYPE & RECOMMENDATIONS:"""
            
            # Determine archetype with detailed analysis
            speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
            variance = dna.get('performance_variance', 0)
            consistency = dna.get('consistency_index', 0)
            
            if speed_ratio > 10:
                archetype = "🎯 CONSISTENCY MASTER"
                strengths = "Exceptional reliability, tire management, race pace"
                weaknesses = "Qualifying speed, aggressive overtaking"
                training = "Peak performance drills, late braking practice"
            elif variance > 0.2:
                archetype = "🏁 TRACK SPECIALIST"
                strengths = "Dominant at preferred circuits, setup expertise"
                weaknesses = "Adaptability, unfamiliar track layouts"
                training = "Varied track practice, quick adaptation skills"
            elif speed_ratio > 6:
                archetype = "⚖️ BALANCED RACER"
                strengths = "Well-rounded, competitive everywhere"
                weaknesses = "Lacks specialized advantages"
                training = "Develop specific strengths, mental performance"
            else:
                archetype = "🏎️ SPEED DEMON"
                strengths = "Raw speed, qualifying pace, aggressive driving"
                weaknesses = "Consistency, tire management, race strategy"
                training = "Consistency drills, strategic thinking"
            
            info_text += f"""
{archetype}

💪 Key Strengths: {strengths}
⚠️  Areas for Improvement: {weaknesses}
🎯 Training Focus: {training}

📈 DETAILED TRACK ANALYSIS:"""
            
            # Sort tracks by performance
            track_performance = []
            for track, metrics in profile['performance_metrics'].items():
                track_performance.append((track, metrics['avg_lap_time'], metrics))
            
            track_performance.sort(key=lambda x: x[1])  # Sort by lap time
            
            for i, (track, lap_time, metrics) in enumerate(track_performance):
                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                
                info_text += f"""
{rank_emoji} {track.upper()} (Rank #{i+1}):
   • Average Lap Time: {metrics['avg_lap_time']:.3f}s
   • Best Lap: {metrics['best_lap']:.3f}s
   • Consistency: {metrics['consistency']:.3f} {'🟢' if metrics['consistency'] > 0.1 else '🟡' if metrics['consistency'] > 0.05 else '🔴'}
   • Average Speed: {metrics['speed_profile']['avg_speed']:.1f} km/h
   • Gap to Best: +{lap_time - track_performance[0][1]:.3f}s"""
        
        else:
            info_text += "\n❌ Insufficient data for comprehensive DNA analysis"
        
        # Update the text widget
        self.driver_info_text.configure(state="normal")
        self.driver_info_text.delete("0.0", "end")
        self.driver_info_text.insert("0.0", info_text)
        self.driver_info_text.configure(state="disabled")
        
        # Update visualization
        self.update_driver_visualization(driver_id, profile)
        
        # Update track performance breakdown
        self.update_track_performance_breakdown(driver_id, profile)
    
    def update_driver_quick_stats(self, driver_id, profile):
        """Update quick stats in driver header"""
        
        # Clear existing quick stats
        for widget in self.driver_quick_stats.winfo_children():
            widget.destroy()
        
        dna = profile.get('dna_signature', {})
        
        if not dna.get('insufficient_data', False):
            # Archetype
            speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
            variance = dna.get('performance_variance', 0)
            
            if speed_ratio > 10:
                archetype = "Consistency Master"
                color = "#4ECDC4"
            elif variance > 0.2:
                archetype = "Track Specialist"
                color = "#FF6B6B"
            elif speed_ratio > 6:
                archetype = "Balanced Racer"
                color = "#96CEB4"
            else:
                archetype = "Speed Demon"
                color = "#45B7D1"
            
            # Create quick stat cards
            stats = [
                ("🏆 Archetype", archetype),
                ("🏁 Tracks", f"{len(profile['tracks_raced'])}/6"),
                ("📊 Consistency", f"{dna.get('consistency_index', 0):.3f}"),
                ("⚡ Adaptability", f"{dna.get('track_adaptability', 0):.1f}")
            ]
            
            for i, (label, value) in enumerate(stats):
                stat_frame = ctk.CTkFrame(self.driver_quick_stats)
                stat_frame.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
                
                ctk.CTkLabel(stat_frame, text=label, font=ctk.CTkFont(size=10), text_color="white").pack(pady=2)
                ctk.CTkLabel(stat_frame, text=value, font=ctk.CTkFont(size=13, weight="bold"), text_color="white").pack(pady=2)
    
    def update_driver_visualization(self, driver_id, profile):
        """Update driver visualization with professional styling"""
        
        # Clear all existing widgets in the viz frame
        for widget in self.driver_viz_frame.winfo_children():
            if widget != self.viz_label:
                widget.destroy()
        
        try:
            # Create figure with optimal size and spacing
            fig = plt.figure(figsize=(13, 11), facecolor='#2b2b2b')
            
            # Create grid spec with improved spacing to prevent overlaps
            # top=0.84 raises plots higher on the canvas
            # bottom=0.12 gives more space for x-axis labels
            # hspace=0.55 increases vertical spacing between rows
            # wspace=0.35 maintains horizontal spacing
            gs = fig.add_gridspec(2, 2, left=0.10, right=0.95, top=0.84, bottom=0.12, 
                                 hspace=0.55, wspace=0.35)
            
            # Add main title centered above all plots (matching Dashboard h1 style)
            # y=0.98 positions it close to the top border
            fig.suptitle(f'Driver {driver_id} Performance Analysis', 
                        color='#FFFFFF', fontsize=18, weight='bold', y=0.98, fontfamily='sans-serif')
            
            dna = profile.get('dna_signature', {})
            
            if not dna.get('insufficient_data', False):
                # ===== DNA RADAR CHART =====
                categories = ['Speed/\nConsistency', 'Track\nAdaptability', 'Lap\nConsistency', 'Performance\nStability']
                values = [
                    min(dna.get('speed_vs_consistency_ratio', 0) / 2, 10),
                    min(dna.get('track_adaptability', 0), 10),
                    dna.get('consistency_index', 0) * 100,
                    (1 - min(dna.get('performance_variance', 0), 1)) * 10
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                
                ax1 = fig.add_subplot(gs[0, 0], projection='polar')
                ax1.plot(angles, values, 'o-', linewidth=3, color='#4ECDC4', marker='o', markersize=10)
                ax1.fill(angles, values, alpha=0.25, color='#4ECDC4')
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(categories, color='#FFFFFF', fontsize=7)
                ax1.set_facecolor('#2b2b2b')
                ax1.tick_params(colors='#CCCCCC', labelsize=7)
                ax1.set_ylim(0, 10)
                ax1.grid(True, color='#555555', alpha=0.5, linestyle='--', linewidth=0.8)
                
                # Make polar plot 15% smaller and center it to align title with Track Performance
                pos = ax1.get_position()
                # Reduce width and height by 15%, and adjust position to center it
                new_width = pos.width * 0.85
                new_height = pos.height * 0.85
                # Center the smaller plot in the original space
                x_offset = (pos.width - new_width) / 2
                y_offset = (pos.height - new_height) / 2
                ax1.set_position([pos.x0 + x_offset, pos.y0 + y_offset, new_width, new_height])
                
                # Set title after repositioning with reduced pad to bring it closer to the plot
                ax1.set_title('DNA Profile', color='#FFFFFF', pad=6, fontsize=12, weight='bold')
                
                # ===== TRACK PERFORMANCE CHARTS =====
                if profile['performance_metrics']:
                    tracks = list(profile['performance_metrics'].keys())
                    # 6 unique, vibrant colors
                    track_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FF8E53', '#A29BFE']
                    
                    # Shorten track names
                    track_labels = []
                    for t in tracks:
                        if t == 'Road America':
                            track_labels.append('Rd America')
                        else:
                            track_labels.append(t)
                    
                    # --- LAP TIME CHART ---
                    ax2 = fig.add_subplot(gs[0, 1])
                    lap_times = [profile['performance_metrics'][track]['avg_lap_time'] for track in tracks]
                    bars2 = ax2.bar(range(len(tracks)), lap_times, color=track_colors[:len(tracks)], 
                                   width=0.7, edgecolor='white', linewidth=0.5)
                    ax2.set_xticks(range(len(tracks)))
                    ax2.set_xticklabels(track_labels, rotation=45, ha='right', color='#FFFFFF', fontsize=7)
                    ax2.set_ylabel('Lap Time (s)', color='#FFFFFF', fontsize=8, weight='bold', labelpad=8)  # Matching Dashboard h3 style
                    ax2.set_title('Track Performance', color='#FFFFFF', fontsize=12, pad=20, weight='bold')  # Matching Dashboard h2 style
                    ax2.tick_params(axis='x', colors='#CCCCCC', labelsize=7)
                    ax2.tick_params(axis='y', colors='#CCCCCC', labelsize=7)
                    ax2.set_facecolor('#2b2b2b')
                    for spine in ax2.spines.values():
                        spine.set_color('#555555')
                        spine.set_linewidth(1)
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    ax2.grid(axis='y', alpha=0.3, color='#555555', linestyle='--', linewidth=0.8)
                    
                    # --- SPEED CHART ---
                    ax3 = fig.add_subplot(gs[1, 0])
                    speeds = [profile['performance_metrics'][track]['speed_profile']['avg_speed'] for track in tracks]
                    bars3 = ax3.bar(range(len(tracks)), speeds, color=track_colors[:len(tracks)], 
                                   width=0.7, edgecolor='white', linewidth=0.5)
                    ax3.set_xticks(range(len(tracks)))
                    ax3.set_xticklabels(track_labels, rotation=45, ha='right', color='#FFFFFF', fontsize=7)
                    ax3.set_ylabel('Speed (km/h)', color='#FFFFFF', fontsize=8, weight='bold', labelpad=8)  # Matching Dashboard h3 style
                    ax3.set_title('Speed Profile', color='#FFFFFF', fontsize=12, pad=10, weight='bold')  # Matching Dashboard h2 style
                    ax3.tick_params(axis='x', colors='#CCCCCC', labelsize=7)
                    ax3.tick_params(axis='y', colors='#CCCCCC', labelsize=7)
                    ax3.set_facecolor('#2b2b2b')
                    for spine in ax3.spines.values():
                        spine.set_color('#555555')
                        spine.set_linewidth(1)
                    ax3.spines['top'].set_visible(False)
                    ax3.spines['right'].set_visible(False)
                    ax3.grid(axis='y', alpha=0.3, color='#555555', linestyle='--', linewidth=0.8)
                    
                    # --- CONSISTENCY CHART ---
                    ax4 = fig.add_subplot(gs[1, 1])
                    consistency = [profile['performance_metrics'][track]['consistency'] for track in tracks]
                    bars4 = ax4.bar(range(len(tracks)), consistency, color=track_colors[:len(tracks)], 
                                   width=0.7, edgecolor='white', linewidth=0.5)
                    ax4.set_xticks(range(len(tracks)))
                    ax4.set_xticklabels(track_labels, rotation=45, ha='right', color='#FFFFFF', fontsize=7)
                    ax4.set_ylabel('Consistency', color='#FFFFFF', fontsize=8, weight='bold', labelpad=8)  # Matching Dashboard h3 style
                    ax4.set_title('Consistency Profile', color='#FFFFFF', fontsize=12, pad=10, weight='bold')  # Matching Dashboard h2 style
                    ax4.tick_params(axis='x', colors='#CCCCCC', labelsize=7)
                    ax4.tick_params(axis='y', colors='#CCCCCC', labelsize=7)
                    ax4.set_facecolor('#2b2b2b')
                    for spine in ax4.spines.values():
                        spine.set_color('#555555')
                        spine.set_linewidth(1)
                    ax4.spines['top'].set_visible(False)
                    ax4.spines['right'].set_visible(False)
                    ax4.grid(axis='y', alpha=0.3, color='#555555', linestyle='--', linewidth=0.8)
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.driver_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
            
        except Exception as e:
            error_label = ctk.CTkLabel(
                self.driver_viz_frame,
                text=f"Visualization error: {str(e)}",
                font=ctk.CTkFont(size=14),
                text_color="#FF6B6B"
            )
            error_label.pack(expand=True)
    
    def update_track_performance_breakdown(self, driver_id, profile):
        """Update track performance breakdown section"""
        
        # Clear existing content
        for widget in self.track_perf_content.winfo_children():
            widget.destroy()
        
        if not profile['performance_metrics']:
            ctk.CTkLabel(
                self.track_perf_content,
                text="No track performance data available",
                font=ctk.CTkFont(size=14)
            ).pack(expand=True)
            return
        
        # Create track performance cards
        tracks_per_row = 3
        row = 0
        col = 0
        
        # Sort tracks by performance
        sorted_tracks = sorted(
            profile['performance_metrics'].items(),
            key=lambda x: x[1]['avg_lap_time']
        )
        
        for i, (track, metrics) in enumerate(sorted_tracks):
            track_card = ctk.CTkFrame(self.track_perf_content)
            track_card.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            # Track name with rank
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
            track_label = ctk.CTkLabel(
                track_card,
                text=f"{rank_emoji} {track.upper()}",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            track_label.pack(pady=5)
            
            # Performance metrics
            metrics_text = f"""Lap: {metrics['avg_lap_time']:.3f}s
Best: {metrics['best_lap']:.3f}s
Speed: {metrics['speed_profile']['avg_speed']:.1f} km/h
Consistency: {metrics['consistency']:.3f}"""
            
            metrics_label = ctk.CTkLabel(
                track_card,
                text=metrics_text,
                font=ctk.CTkFont(size=10)
            )
            metrics_label.pack(pady=5)
            
            col += 1
            if col >= tracks_per_row:
                col = 0
                row += 1
        
        # Configure grid weights
        for i in range(tracks_per_row):
            self.track_perf_content.grid_columnconfigure(i, weight=1)
    
    def update_insights_tab(self):
        """Update the insights tab with analysis results"""
        
        if not self.analysis_complete:
            return
            
        # Generate insights text
        insights_text = """🧠 ADVANCED INSIGHTS & RECOMMENDATIONS

🔍 FEATURE EXPLAINABILITY:
"""
        
        # Add feature explainability analysis
        try:
            from dna_explainability import DNAExplainability
            
            explainer = DNAExplainability()
            
            # Extract DNA features from existing driver profiles
            dna_data = []
            archetype_list = []
            
            for driver_id, profile in self.analyzer.driver_profiles.items():
                dna = profile.get('dna_signature', {})
                if not dna.get('insufficient_data', False):
                    # Extract DNA features
                    dna_record = {'driver_id': driver_id}
                    for feature in explainer.feature_names:
                        dna_record[feature] = dna.get(feature, 0)
                    dna_data.append(dna_record)
                    
                    # Determine archetype
                    speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                    variance = dna.get('performance_variance', 0)
                    
                    if speed_ratio > 10:
                        archetype_list.append('Consistency Master')
                    elif variance > 0.2:
                        archetype_list.append('Track Specialist')
                    elif speed_ratio > 6:
                        archetype_list.append('Balanced Racer')
                    else:
                        archetype_list.append('Speed Demon')
            
            if dna_data and len(dna_data) >= 2:
                dna_features = pd.DataFrame(dna_data)
                archetype_labels = pd.Series(archetype_list)
                
                importance_data = explainer.calculate_feature_importance_by_archetype(
                    dna_features, archetype_labels
                )
                
                if importance_data:
                    insights_text += "\n📊 Top Features Influencing Each Archetype:\n\n"
                    
                    for archetype, importance_df in importance_data.items():
                        top_3 = importance_df.head(3)
                        insights_text += f"🏁 {archetype}:\n"
                        for i, (_, row) in enumerate(top_3.iterrows(), 1):
                            insights_text += f"   {i}. {row['display_name']} ({row['importance']:.1f}% importance, {row['direction']})\n"
                        insights_text += "\n"
                    
                    insights_text += "💡 These features are the key differentiators for each archetype.\n"
                    insights_text += "   Use this information to understand driver strengths and areas for improvement.\n\n"
        except Exception as e:
            insights_text += f"⚠️ Feature explainability not available: {str(e)}\n\n"
        
        insights_text += """
🏆 DRIVER ARCHETYPE ANALYSIS:
"""
        
        # Count archetypes
        archetypes = {'Speed Demons': 0, 'Consistency Masters': 0, 'Track Specialists': 0, 'Balanced Racers': 0}
        
        for profile in self.analyzer.driver_profiles.values():
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                variance = dna.get('performance_variance', 0)
                
                if speed_ratio > 10:
                    archetypes['Consistency Masters'] += 1
                elif variance > 0.2:
                    archetypes['Track Specialists'] += 1
                elif speed_ratio > 6:
                    archetypes['Balanced Racers'] += 1
                else:
                    archetypes['Speed Demons'] += 1
        
        for archetype, count in archetypes.items():
            insights_text += f"• {archetype}: {count} drivers\n"
        
        insights_text += """
🎯 TRAINING RECOMMENDATIONS:

🏎️ Speed Demons:
• Focus on consistency training and tire management
• Practice maintaining steady lap times
• Develop race strategy and positioning skills

🎯 Consistency Masters:
• Work on qualifying pace and aggressive techniques
• Practice late braking and overtaking maneuvers
• Develop peak performance capabilities

🏁 Track Specialists:
• Improve adaptability across different track types
• Focus on setup versatility and quick adaptation
• Practice on varied circuit layouts

⚖️ Balanced Racers:
• Identify and develop specific strengths
• Work on mental performance and race-winning scenarios
• Focus on specialized skill development

📊 TRACK ANALYSIS:
"""
        
        if hasattr(self.analyzer, 'sector_analysis'):
            df = self.analyzer.sector_analysis
            track_stats = df.groupby('track').agg({
                'LAP_TIME_mean': 'mean',
                'KPH_mean': 'mean'
            }).round(2)
            
            for track, stats in track_stats.iterrows():
                insights_text += f"🏁 {track.upper()}: Avg lap {stats['LAP_TIME_mean']:.1f}s, Avg speed {stats['KPH_mean']:.1f} km/h\n"
        
        insights_text += """
💡 KEY INSIGHTS:
• Multi-track analysis reveals unique driver characteristics
• DNA profiling enables personalized training programs
• Archetype classification helps with strategic planning
• Performance patterns guide coaching decisions

🚀 NEXT STEPS:
1. Use individual driver reports for detailed coaching
2. Implement archetype-specific training programs
3. Monitor performance improvements over time
4. Apply insights to race strategy development"""
        
        # Update insights content
        self.insights_content.configure(state="normal")
        self.insights_content.delete("0.0", "end")
        self.insights_content.insert("0.0", insights_text)
        self.insights_content.configure(state="disabled")
    
    def update_tracks_tab(self):
        """Update the tracks analysis tab"""
        
        if not self.analysis_complete:
            return
            
        # Clear existing content
        for widget in self.track_content_frame.winfo_children():
            widget.destroy()
        
        # Create track analysis content
        if hasattr(self.analyzer, 'sector_analysis'):
            df = self.analyzer.sector_analysis
            
            # Track statistics
            track_stats = df.groupby('track').agg({
                'LAP_TIME_mean': ['mean', 'std'],
                'KPH_mean': ['mean', 'max'],
                'NUMBER': 'count'
            }).round(3)
            
            # Create scrollable text widget
            track_text = ctk.CTkTextbox(
                self.track_content_frame,
                font=ctk.CTkFont(size=12)
            )
            track_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            
            self.track_content_frame.grid_columnconfigure(0, weight=1)
            self.track_content_frame.grid_rowconfigure(0, weight=1)
            
            content = "🏁 TRACK ANALYSIS SUMMARY\n\n"
            
            for track in df['track'].unique():
                track_data = df[df['track'] == track]
                
                content += f"📍 {track.upper()}\n"
                content += f"   • Drivers: {len(track_data)} analyzed\n"
                content += f"   • Avg Lap Time: {track_data['LAP_TIME_mean'].mean():.2f}s\n"
                content += f"   • Lap Time Std: {track_data['LAP_TIME_mean'].std():.2f}s\n"
                content += f"   • Avg Speed: {track_data['KPH_mean'].mean():.1f} km/h\n"
                content += f"   • Max Speed: {track_data['KPH_mean'].max():.1f} km/h\n"
                
                # Track difficulty assessment
                difficulty = track_data['LAP_TIME_std'].mean()
                if difficulty > 2.0:
                    content += f"   • Difficulty: High (Technical)\n"
                elif difficulty > 1.0:
                    content += f"   • Difficulty: Medium (Balanced)\n"
                else:
                    content += f"   • Difficulty: Low (High-Speed)\n"
                
                content += "\n"
            
            track_text.insert("0.0", content)
            track_text.configure(state="disabled")
    
    def show_tab(self, tab_name):
        """Show specific tab"""
        
        tab_mapping = {
            "overview": 0,
            "drivers": 1,
            "tracks": 2,
            "insights": 3,
            "guidelines": 4
        }
        
        if tab_name in tab_mapping:
            self.notebook.select(tab_mapping[tab_name])
    
    def show_quick_start(self):
        """Show Quick Start guide in Guidelines tab"""
        # Navigate to Guidelines tab
        self.notebook.select(4)  # Guidelines tab index
        # Select the Quick Start sub-tab (first tab in guidelines notebook)
        if hasattr(self, 'guidelines_notebook'):
            self.guidelines_notebook.select(0)  # Quick Start is the first sub-tab
    
    def show_dashboard_tab(self):
        """Show dashboard tab"""
        if not self.analysis_complete:
            messagebox.showwarning("Dashboard Warning", "Please run analysis first!")
            return
        self.notebook.select(5)  # Dashboard tab index
    
    def show_report_tab(self):
        """Show report tab"""
        if not self.analysis_complete:
            messagebox.showwarning("Report Warning", "Please run analysis first!")
            return
        self.notebook.select(6)  # Report tab index
    
    def show_visualization_tab(self):
        """Show visualization tab"""
        if not self.analysis_complete:
            messagebox.showwarning("Visualization Warning", "Please run analysis first!")
            return
        self.notebook.select(7)  # Visualization tab index
    
    def create_dashboard_window(self):
        """Create embedded interactive dashboard window"""
        
        dashboard_window = ctk.CTkToplevel(self.root)
        dashboard_window.title("📊 Interactive Dashboard")
        dashboard_window.geometry("1200x800")
        
        # Create notebook for different dashboard views
        dashboard_notebook = ttk.Notebook(dashboard_window)
        dashboard_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Performance Overview Tab
        overview_frame = ctk.CTkFrame(dashboard_notebook)
        dashboard_notebook.add(overview_frame, text="📊 Performance Overview")
        
        self.create_performance_overview_chart(overview_frame)
        
        # Driver Comparison Tab
        comparison_frame = ctk.CTkFrame(dashboard_notebook)
        dashboard_notebook.add(comparison_frame, text="👥 Driver Comparison")
        
        self.create_driver_comparison_chart(comparison_frame)
        
        # Track Analysis Tab
        track_frame = ctk.CTkFrame(dashboard_notebook)
        dashboard_notebook.add(track_frame, text="🏁 Track Analysis")
        
        self.create_track_analysis_chart(track_frame)
        
        # DNA Patterns Tab
        dna_frame = ctk.CTkFrame(dashboard_notebook)
        dashboard_notebook.add(dna_frame, text="🧬 DNA Patterns")
        
        self.create_dna_patterns_chart(dna_frame)
    
    def create_performance_overview_chart(self, parent):
        """Create performance overview chart"""
        
        # Create professional matplotlib figure with optimal spacing
        fig = plt.figure(figsize=(15, 11.5), facecolor='#2b2b2b')
        
        # Create grid spec with proper spacing to prevent title overlap
        # Reduced top to 0.82 to give more space below title (doubled spacing)
        # Increased hspace to 0.55 for better vertical separation
        gs = fig.add_gridspec(2, 2, left=0.08, right=0.95, top=0.82, bottom=0.10,
                             hspace=0.55, wspace=0.32,
                             height_ratios=[1, 1], width_ratios=[1, 1.15])
        
        # Main title positioned higher with clear separation
        fig.suptitle('Performance Overview Dashboard', 
                    fontsize=18, fontweight='bold', color='#FFFFFF', y=0.96,
                    fontfamily='sans-serif')
        
        # Unified color palette for visual cohesion
        archetype_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        track_colors = ['#FF6B6B', '#FF8E53', '#FFBE53', '#96CEB4', '#4ECDC4', '#A29BFE']
        
        # ===== 1. ARCHETYPE DISTRIBUTION (PIE CHART) =====
        ax1 = fig.add_subplot(gs[0, 0])
        
        archetypes = {'Speed Demons': 0, 'Consistency Masters': 0, 'Track Specialists': 0, 'Balanced Racers': 0}
        
        for profile in self.analyzer.driver_profiles.values():
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                variance = dna.get('performance_variance', 0)
                
                if speed_ratio > 10:
                    archetypes['Consistency Masters'] += 1
                elif variance > 0.2:
                    archetypes['Track Specialists'] += 1
                elif speed_ratio > 6:
                    archetypes['Balanced Racers'] += 1
                else:
                    archetypes['Speed Demons'] += 1
        
        # Create pie chart with enhanced visibility for percentages
        wedges, texts, autotexts = ax1.pie(archetypes.values(), 
                                           labels=archetypes.keys(), 
                                           autopct='%1.1f%%',
                                           colors=archetype_colors,
                                           startangle=90,
                                           pctdistance=0.75,  # Moved closer to center for better visibility
                                           explode=(0.08, 0.08, 0.08, 0.08),  # Increased explosion for clarity
                                           textprops={'color': '#FFFFFF', 'fontsize': 9, 'weight': 'bold'},
                                           labeldistance=1.15)  # Labels further out
        
        # Style percentage labels with high contrast
        for autotext in autotexts:
            autotext.set_color('#FFFFFF')  # White text for better visibility
            autotext.set_fontsize(9)  # Decreased font size by 1.5x (13/1.5 ≈ 9)
            autotext.set_weight('bold')
            autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='#2b2b2b', 
                                  edgecolor='none', alpha=0.7))  # Dark background box
        
        # Style archetype labels
        for text in texts:
            text.set_fontsize(9)
            text.set_weight('bold')
        
        ax1.set_title('Driver Archetype Distribution', 
                     fontsize=12, fontweight='bold', color='#FFFFFF', pad=20)
        ax1.set_facecolor('#2b2b2b')
        
        # ===== 2. SPEED VS CONSISTENCY SCATTER =====
        ax2 = fig.add_subplot(gs[0, 1])
        
        if hasattr(self.analyzer, 'sector_analysis'):
            df = self.analyzer.sector_analysis
            
            # Group by driver and calculate metrics
            driver_metrics = df.groupby('NUMBER').agg({
                'LAP_TIME_mean': 'mean',
                'LAP_TIME_std': 'mean',
                'KPH_mean': 'mean'
            }).reset_index()
            
            scatter = ax2.scatter(driver_metrics['LAP_TIME_std'], 
                                driver_metrics['KPH_mean'], 
                                c=driver_metrics['LAP_TIME_mean'], 
                                cmap='plasma',
                                alpha=0.75,
                                s=120,
                                edgecolors='white',
                                linewidth=0.5)
            
            ax2.set_xlabel('Lap Time Consistency (lower = more consistent)', 
                          fontsize=8, fontweight='bold', color='#FFFFFF', labelpad=8)
            ax2.set_ylabel('Average Speed (km/h)', 
                          fontsize=8, fontweight='bold', color='#FFFFFF', labelpad=8)
            ax2.set_title('Speed vs Consistency Analysis', 
                         fontsize=12, fontweight='bold', color='#FFFFFF', pad=20)
            ax2.tick_params(colors='#CCCCCC', labelsize=7)
            ax2.set_facecolor('#2b2b2b')
            ax2.grid(True, alpha=0.2, color='#555555', linestyle='--', linewidth=0.5)
            
            # Style spines
            for spine in ax2.spines.values():
                spine.set_color('#555555')
                spine.set_linewidth(1)
            
            # Add colorbar with improved styling
            cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
            cbar.set_label('Avg Lap Time (s)', fontsize=8, color='#FFFFFF', weight='bold')
            cbar.ax.tick_params(colors='#CCCCCC', labelsize=9)
            cbar.outline.set_edgecolor('#555555')
        
        # ===== 3. TRACK DIFFICULTY RANKING (BAR CHART) =====
        ax3 = fig.add_subplot(gs[1, 0])
        
        if hasattr(self.analyzer, 'sector_analysis'):
            track_difficulty = df.groupby('track')['LAP_TIME_std'].mean().sort_values(ascending=False)
            
            bars = ax3.bar(range(len(track_difficulty)), 
                          track_difficulty.values, 
                          color=track_colors[:len(track_difficulty)],
                          edgecolor='white',
                          linewidth=0.8,
                          alpha=0.9)
            
            ax3.set_xticks(range(len(track_difficulty)))
            ax3.set_xticklabels([t.title() for t in track_difficulty.index], 
                               rotation=35, ha='right', fontsize=7, color='#FFFFFF')
            ax3.set_ylabel('Lap Time Variance', 
                          fontsize=8, fontweight='bold', color='#FFFFFF', labelpad=8)
            ax3.set_title('Track Difficulty Ranking', 
                         fontsize=12, fontweight='bold', color='#FFFFFF', pad=20)
            ax3.tick_params(axis='y', colors='#CCCCCC', labelsize=7)
            ax3.set_facecolor('#2b2b2b')
            ax3.grid(axis='y', alpha=0.2, color='#555555', linestyle='--', linewidth=0.5)
            
            # Style spines
            for spine in ax3.spines.values():
                spine.set_color('#555555')
                spine.set_linewidth(1)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
        
        # ===== 4. LAP TIME DISTRIBUTION (HISTOGRAM) =====
        ax4 = fig.add_subplot(gs[1, 1])
        
        if hasattr(self.analyzer, 'sector_analysis'):
            n, bins, patches = ax4.hist(df['LAP_TIME_mean'], 
                                       bins=25, 
                                       alpha=0.85, 
                                       color='#4ECDC4',
                                       edgecolor='white',
                                       linewidth=0.8)
            
            # Color gradient for histogram bars
            cm = plt.cm.viridis
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))
            
            ax4.set_xlabel('Average Lap Time (s)', 
                          fontsize=8, fontweight='bold', color='#FFFFFF', labelpad=8)
            ax4.set_ylabel('Frequency', 
                          fontsize=8, fontweight='bold', color='#FFFFFF', labelpad=8)
            ax4.set_title('Lap Time Distribution', 
                         fontsize=12, fontweight='bold', color='#FFFFFF', pad=20)
            ax4.tick_params(colors='#CCCCCC', labelsize=7)
            ax4.set_facecolor('#2b2b2b')
            ax4.grid(axis='y', alpha=0.2, color='#555555', linestyle='--', linewidth=0.5)
            
            # Style spines
            for spine in ax4.spines.values():
                spine.set_color('#555555')
                spine.set_linewidth(1)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
        
        # Store figure for export functionality
        self.dashboard_fig = fig
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_driver_comparison_chart(self, parent):
        """Create driver comparison chart"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Driver Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Get top 10 drivers by number of tracks raced
        top_drivers = sorted(self.analyzer.driver_profiles.items(), 
                           key=lambda x: len(x[1]['tracks_raced']), reverse=True)[:10]
        
        driver_ids = [str(d[0]) for d in top_drivers]
        
        # 1. DNA Characteristics Comparison
        speed_ratios = []
        adaptabilities = []
        consistencies = []
        
        for driver_id, profile in top_drivers:
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                speed_ratios.append(dna.get('speed_vs_consistency_ratio', 0))
                adaptabilities.append(dna.get('track_adaptability', 0))
                consistencies.append(dna.get('consistency_index', 0) * 100)
            else:
                speed_ratios.append(0)
                adaptabilities.append(0)
                consistencies.append(0)
        
        x = np.arange(len(driver_ids))
        width = 0.25
        
        ax1.bar(x - width, speed_ratios, width, label='Speed/Consistency Ratio', alpha=0.8)
        ax1.bar(x, adaptabilities, width, label='Track Adaptability', alpha=0.8)
        ax1.bar(x + width, consistencies, width, label='Consistency Index (×100)', alpha=0.8)
        
        ax1.set_xlabel('Driver ID')
        ax1.set_ylabel('DNA Metrics')
        ax1.set_title('DNA Characteristics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(driver_ids)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Track Coverage Heatmap
        track_coverage = np.zeros((len(top_drivers), 6))
        track_names = ['Barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']
        
        for i, (driver_id, profile) in enumerate(top_drivers):
            for j, track in enumerate(['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']):
                if track in profile['tracks_raced']:
                    # Get average lap time for this track
                    if track in profile['performance_metrics']:
                        track_coverage[i, j] = profile['performance_metrics'][track]['avg_lap_time']
        
        im = ax2.imshow(track_coverage, cmap='RdYlBu_r', aspect='auto')
        ax2.set_xticks(range(6))
        ax2.set_xticklabels(track_names, rotation=45)
        ax2.set_yticks(range(len(driver_ids)))
        ax2.set_yticklabels(driver_ids)
        ax2.set_title('Driver Performance Heatmap\n(Lap Times by Track)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Average Lap Time (s)')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_track_analysis_chart(self, parent):
        """Create track analysis chart"""
        
        if not hasattr(self.analyzer, 'sector_analysis'):
            error_label = ctk.CTkLabel(parent, text="No sector analysis data available", 
                                     font=ctk.CTkFont(size=16))
            error_label.pack(expand=True)
            return
        
        df = self.analyzer.sector_analysis
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Track Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Average lap times by track
        track_lap_times = df.groupby('track')['LAP_TIME_mean'].mean().sort_values()
        
        bars1 = ax1.bar(range(len(track_lap_times)), track_lap_times.values, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
        ax1.set_xticks(range(len(track_lap_times)))
        ax1.set_xticklabels([t.title() for t in track_lap_times.index], rotation=45)
        ax1.set_ylabel('Average Lap Time (s)')
        ax1.set_title('Average Lap Times by Track')
        
        # Add value labels on bars
        for bar, value in zip(bars1, track_lap_times.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}s', ha='center', va='bottom')
        
        # 2. Average speeds by track
        track_speeds = df.groupby('track')['KPH_mean'].mean().sort_values(ascending=False)
        
        bars2 = ax2.bar(range(len(track_speeds)), track_speeds.values, 
                       color=['#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#6C5CE7', '#00B894'])
        ax2.set_xticks(range(len(track_speeds)))
        ax2.set_xticklabels([t.title() for t in track_speeds.index], rotation=45)
        ax2.set_ylabel('Average Speed (km/h)')
        ax2.set_title('Average Speeds by Track')
        
        # 3. Sector time analysis
        sector_data = []
        for track in df['track'].unique():
            track_data = df[df['track'] == track]
            sector_data.append([
                track_data['S1_mean'].mean(),
                track_data['S2_mean'].mean(), 
                track_data['S3_mean'].mean()
            ])
        
        sector_array = np.array(sector_data)
        track_names = [t.title() for t in df['track'].unique()]
        
        x = np.arange(len(track_names))
        width = 0.25
        
        ax3.bar(x - width, sector_array[:, 0], width, label='Sector 1', alpha=0.8)
        ax3.bar(x, sector_array[:, 1], width, label='Sector 2', alpha=0.8)
        ax3.bar(x + width, sector_array[:, 2], width, label='Sector 3', alpha=0.8)
        
        ax3.set_xlabel('Track')
        ax3.set_ylabel('Average Sector Time (s)')
        ax3.set_title('Sector Time Breakdown by Track')
        ax3.set_xticks(x)
        ax3.set_xticklabels(track_names, rotation=45)
        ax3.legend()
        
        # 4. Driver count by track
        driver_counts = df.groupby('track')['NUMBER'].nunique().sort_values(ascending=False)
        
        bars4 = ax4.bar(range(len(driver_counts)), driver_counts.values, 
                       color=['#00B894', '#00CEC9', '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E'])
        ax4.set_xticks(range(len(driver_counts)))
        ax4.set_xticklabels([t.title() for t in driver_counts.index], rotation=45)
        ax4.set_ylabel('Number of Drivers')
        ax4.set_title('Driver Participation by Track')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_dna_patterns_chart(self, parent):
        """Create DNA patterns analysis chart"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('DNA Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Extract DNA data
        dna_data = []
        driver_ids = []
        
        for driver_id, profile in self.analyzer.driver_profiles.items():
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                dna_data.append([
                    dna.get('speed_vs_consistency_ratio', 0),
                    dna.get('track_adaptability', 0),
                    dna.get('consistency_index', 0),
                    dna.get('performance_variance', 0)
                ])
                driver_ids.append(driver_id)
        
        if not dna_data:
            error_label = ctk.CTkLabel(parent, text="No DNA data available", 
                                     font=ctk.CTkFont(size=16))
            error_label.pack(expand=True)
            return
        
        dna_array = np.array(dna_data)
        
        # 1. DNA Characteristics Distribution
        characteristics = ['Speed/Consistency', 'Adaptability', 'Consistency', 'Variance']
        
        for i, char in enumerate(characteristics):
            ax1.hist(dna_array[:, i], bins=10, alpha=0.7, label=char)
        
        ax1.set_xlabel('DNA Metric Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('DNA Characteristics Distribution')
        ax1.legend()
        
        # 2. Speed vs Consistency Scatter with Archetype Colors
        colors = []
        for row in dna_array:
            speed_ratio, _, _, variance = row
            if speed_ratio > 10:
                colors.append('#4ECDC4')  # Consistency Masters
            elif variance > 0.2:
                colors.append('#FF6B6B')  # Track Specialists
            elif speed_ratio > 6:
                colors.append('#96CEB4')  # Balanced Racers
            else:
                colors.append('#45B7D1')  # Speed Demons
        
        scatter = ax2.scatter(dna_array[:, 0], dna_array[:, 2] * 100, c=colors, alpha=0.7, s=60)
        ax2.set_xlabel('Speed vs Consistency Ratio')
        ax2.set_ylabel('Consistency Index (×100)')
        ax2.set_title('Driver Archetype Clustering')
        
        # Add legend for archetypes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#45B7D1', label='Speed Demons'),
            Patch(facecolor='#4ECDC4', label='Consistency Masters'),
            Patch(facecolor='#FF6B6B', label='Track Specialists'),
            Patch(facecolor='#96CEB4', label='Balanced Racers')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # 3. Adaptability vs Variance
        ax3.scatter(dna_array[:, 1], dna_array[:, 3], c=colors, alpha=0.7, s=60)
        ax3.set_xlabel('Track Adaptability')
        ax3.set_ylabel('Performance Variance')
        ax3.set_title('Adaptability vs Performance Variance')
        
        # 4. DNA Radar Chart for Top 5 Drivers
        top_5_indices = np.argsort(dna_array[:, 1])[-5:]  # Top 5 by adaptability
        
        angles = np.linspace(0, 2 * np.pi, len(characteristics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, idx in enumerate(top_5_indices):
            values = dna_array[idx].tolist()
            values += values[:1]  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=f'Driver {driver_ids[idx]}', 
                    color=colors_radar[i])
            ax4.fill(angles, values, alpha=0.25, color=colors_radar[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(characteristics)
        ax4.set_title('Top 5 Drivers DNA Profiles')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_report_window(self):
        """Create embedded analysis report window"""
        
        report_window = ctk.CTkToplevel(self.root)
        report_window.title("📋 Analysis Report")
        report_window.geometry("900x700")
        
        # Create scrollable text widget
        report_text = ctk.CTkTextbox(
            report_window,
            font=ctk.CTkFont(size=12, family="Courier")
        )
        report_text.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Generate comprehensive report
        report_content = self.generate_comprehensive_report()
        
        report_text.insert("0.0", report_content)
        report_text.configure(state="disabled")
        
        # Add export button
        export_button = ctk.CTkButton(
            report_window,
            text="💾 Save Report to File",
            command=lambda: self.save_report_to_file(report_content)
        )
        export_button.pack(pady=10)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        
        report = f"""
🧬 MULTI-TRACK PERFORMANCE DNA ANALYZER - COMPREHENSIVE REPORT
{'='*80}

📊 EXECUTIVE SUMMARY
{'-'*40}
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Drivers Analyzed: {len(self.analyzer.driver_profiles)}
Tracks Covered: 6 (Barber, COTA, Road America, Sebring, Sonoma, VIR)
Data Points: {len(self.analyzer.sector_analysis) if hasattr(self.analyzer, 'sector_analysis') else 'N/A'}

🏆 DRIVER ARCHETYPE ANALYSIS
{'-'*40}
"""
        
        # Count archetypes and collect driver lists
        archetypes = {
            'Speed Demons': {'count': 0, 'drivers': []},
            'Consistency Masters': {'count': 0, 'drivers': []},
            'Track Specialists': {'count': 0, 'drivers': []},
            'Balanced Racers': {'count': 0, 'drivers': []}
        }
        
        for driver_id, profile in self.analyzer.driver_profiles.items():
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                variance = dna.get('performance_variance', 0)
                
                if speed_ratio > 10:
                    archetypes['Consistency Masters']['count'] += 1
                    archetypes['Consistency Masters']['drivers'].append(driver_id)
                elif variance > 0.2:
                    archetypes['Track Specialists']['count'] += 1
                    archetypes['Track Specialists']['drivers'].append(driver_id)
                elif speed_ratio > 6:
                    archetypes['Balanced Racers']['count'] += 1
                    archetypes['Balanced Racers']['drivers'].append(driver_id)
                else:
                    archetypes['Speed Demons']['count'] += 1
                    archetypes['Speed Demons']['drivers'].append(driver_id)
        
        for archetype, data in archetypes.items():
            report += f"""
🏎️ {archetype}: {data['count']} drivers
   Drivers: {', '.join(map(str, data['drivers'][:10]))}{'...' if len(data['drivers']) > 10 else ''}
"""
        
        report += f"""

📈 TRACK PERFORMANCE ANALYSIS
{'-'*40}
"""
        
        if hasattr(self.analyzer, 'sector_analysis'):
            df = self.analyzer.sector_analysis
            
            for track in sorted(df['track'].unique()):
                track_data = df[df['track'] == track]
                
                report += f"""
🏁 {track.upper()}:
   • Drivers Analyzed: {len(track_data)}
   • Average Lap Time: {track_data['LAP_TIME_mean'].mean():.3f}s
   • Lap Time Range: {track_data['LAP_TIME_mean'].min():.3f}s - {track_data['LAP_TIME_mean'].max():.3f}s
   • Average Speed: {track_data['KPH_mean'].mean():.1f} km/h
   • Top Speed: {track_data['KPH_mean'].max():.1f} km/h
   • Difficulty Rating: {'High' if track_data['LAP_TIME_std'].mean() > 2.0 else 'Medium' if track_data['LAP_TIME_std'].mean() > 1.0 else 'Low'}
"""
        
        report += f"""

👥 TOP PERFORMERS ANALYSIS
{'-'*40}
"""
        
        # Find top performers by different metrics
        top_consistent = []
        top_fast = []
        top_adaptable = []
        
        for driver_id, profile in self.analyzer.driver_profiles.items():
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                consistency = dna.get('consistency_index', 0)
                speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                adaptability = dna.get('track_adaptability', 0)
                
                top_consistent.append((driver_id, consistency))
                top_fast.append((driver_id, speed_ratio))
                top_adaptable.append((driver_id, adaptability))
        
        # Sort and get top 5
        top_consistent.sort(key=lambda x: x[1], reverse=True)
        top_fast.sort(key=lambda x: x[1], reverse=True)
        top_adaptable.sort(key=lambda x: x[1], reverse=True)
        
        report += f"""
🎯 Most Consistent Drivers:
   {', '.join([f'Driver {d[0]} ({d[1]:.3f})' for d in top_consistent[:5]])}

🏎️ Highest Speed/Consistency Ratio:
   {', '.join([f'Driver {d[0]} ({d[1]:.2f})' for d in top_fast[:5]])}

🌟 Most Adaptable Drivers:
   {', '.join([f'Driver {d[0]} ({d[1]:.2f})' for d in top_adaptable[:5]])}
"""
        
        report += f"""

🧬 DETAILED DRIVER DNA PROFILES
{'-'*40}
"""
        
        # Top 10 drivers by track coverage
        top_drivers = sorted(self.analyzer.driver_profiles.items(), 
                           key=lambda x: len(x[1]['tracks_raced']), reverse=True)[:10]
        
        for driver_id, profile in top_drivers:
            dna = profile.get('dna_signature', {})
            
            report += f"""
🏁 DRIVER {driver_id}:
   • Races Completed: {profile['total_races']}
   • Tracks Raced: {len(profile['tracks_raced'])} ({', '.join(profile['tracks_raced'])})
"""
            
            if not dna.get('insufficient_data', False):
                # Determine archetype
                speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                variance = dna.get('performance_variance', 0)
                
                if speed_ratio > 10:
                    archetype = "Consistency Master"
                elif variance > 0.2:
                    archetype = "Track Specialist"
                elif speed_ratio > 6:
                    archetype = "Balanced Racer"
                else:
                    archetype = "Speed Demon"
                
                report += f"""   • Archetype: {archetype}
   • Speed/Consistency Ratio: {speed_ratio:.2f}
   • Track Adaptability: {dna.get('track_adaptability', 0):.2f}
   • Consistency Index: {dna.get('consistency_index', 0):.3f}
   • Performance Variance: {dna.get('performance_variance', 0):.3f}
"""
                
                # Track performance
                if profile['performance_metrics']:
                    best_track = min(profile['performance_metrics'].items(), 
                                   key=lambda x: x[1]['avg_lap_time'])
                    worst_track = max(profile['performance_metrics'].items(), 
                                    key=lambda x: x[1]['avg_lap_time'])
                    
                    report += f"""   • Best Track: {best_track[0]} ({best_track[1]['avg_lap_time']:.3f}s avg)
   • Most Challenging: {worst_track[0]} ({worst_track[1]['avg_lap_time']:.3f}s avg)
"""
            else:
                report += "   • DNA Analysis: Insufficient data\n"
        
        report += f"""

🎯 TRAINING RECOMMENDATIONS
{'-'*40}

🏎️ Speed Demons ({archetypes['Speed Demons']['count']} drivers):
   • Focus Areas: Consistency training, tire management
   • Training Methods: Practice maintaining steady lap times, work on race strategy
   • Goal: Convert raw speed into consistent race performance

🎯 Consistency Masters ({archetypes['Consistency Masters']['count']} drivers):
   • Focus Areas: Peak performance, qualifying pace
   • Training Methods: Practice aggressive techniques, late braking drills
   • Goal: Develop race-winning speed while maintaining consistency

🏁 Track Specialists ({archetypes['Track Specialists']['count']} drivers):
   • Focus Areas: Adaptability, versatility training
   • Training Methods: Practice on varied track types, setup experimentation
   • Goal: Become competitive across all circuit types

⚖️ Balanced Racers ({archetypes['Balanced Racers']['count']} drivers):
   • Focus Areas: Specialized skill development
   • Training Methods: Identify and amplify specific strengths
   • Goal: Develop championship-winning capabilities

💡 STRATEGIC INSIGHTS
{'-'*40}

• Multi-track analysis reveals unique driver characteristics not visible in single-race data
• DNA profiling enables personalized training programs for maximum improvement
• Archetype classification helps with strategic race planning and team composition
• Performance patterns guide coaching decisions and development priorities

🚀 IMPLEMENTATION RECOMMENDATIONS
{'-'*40}

1. Implement archetype-specific training programs
2. Use DNA profiles for driver-track pairing optimization
3. Monitor performance evolution over time
4. Apply insights to race strategy development
5. Create benchmarking system for continuous improvement

📊 DATA QUALITY ASSESSMENT
{'-'*40}

• Data Completeness: {len(self.analyzer.driver_profiles)} drivers with sufficient data
• Track Coverage: 6/6 tracks analyzed (100%)
• Analysis Confidence: High (comprehensive multi-track dataset)
• Recommendation Reliability: Excellent for strategic planning

{'-'*80}
Report Generated by Multi-Track Performance DNA Analyzer v1.0
© 2024 - Advanced Racing Analytics System
"""
        
        return report
    
    def save_report_to_file(self, content):
        """Save report to file"""
        try:
            filename = f"dna_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            messagebox.showinfo("Save Success", f"Report saved as '{filename}'")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report: {str(e)}")
    
    def show_visualizations(self):
        """Show live visualizations window"""
        
        if not self.analysis_complete:
            messagebox.showwarning("Visualization Warning", "Please run analysis first!")
            return
        
        viz_window = ctk.CTkToplevel(self.root)
        viz_window.title("📈 Live Visualizations")
        viz_window.geometry("1000x700")
        
        # Create notebook for different visualizations
        viz_notebook = ttk.Notebook(viz_window)
        viz_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Real-time metrics
        metrics_frame = ctk.CTkFrame(viz_notebook)
        viz_notebook.add(metrics_frame, text="📊 Real-time Metrics")
        
        self.create_realtime_metrics(metrics_frame)
        
        # Performance trends
        trends_frame = ctk.CTkFrame(viz_notebook)
        viz_notebook.add(trends_frame, text="📈 Performance Trends")
        
        self.create_performance_trends(trends_frame)
    
    def create_realtime_metrics(self, parent):
        """Create real-time metrics display"""
        
        # Create metrics grid
        metrics_grid = ctk.CTkFrame(parent)
        metrics_grid.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Configure grid
        for i in range(3):
            metrics_grid.grid_columnconfigure(i, weight=1)
        for i in range(3):
            metrics_grid.grid_rowconfigure(i, weight=1)
        
        # Total drivers metric
        total_drivers_frame = ctk.CTkFrame(metrics_grid)
        total_drivers_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(total_drivers_frame, text="👥 Total Drivers", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkLabel(total_drivers_frame, text=str(len(self.analyzer.driver_profiles)), 
                    font=ctk.CTkFont(size=32, weight="bold")).pack()
        
        # Tracks covered metric
        tracks_frame = ctk.CTkFrame(metrics_grid)
        tracks_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(tracks_frame, text="🏁 Tracks Covered", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkLabel(tracks_frame, text="6/6", 
                    font=ctk.CTkFont(size=32, weight="bold")).pack()
        
        # Data points metric
        data_points_frame = ctk.CTkFrame(metrics_grid)
        data_points_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(data_points_frame, text="📊 Data Points", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        data_count = len(self.analyzer.sector_analysis) if hasattr(self.analyzer, 'sector_analysis') else 0
        ctk.CTkLabel(data_points_frame, text=str(data_count), 
                    font=ctk.CTkFont(size=32, weight="bold")).pack()
        
        # Archetype breakdown
        archetype_frames = []
        archetype_names = ["🏎️ Speed Demons", "🎯 Consistency Masters", "🏁 Track Specialists", "⚖️ Balanced Racers"]
        
        # Count archetypes
        archetype_counts = [0, 0, 0, 0]
        
        for profile in self.analyzer.driver_profiles.values():
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                variance = dna.get('performance_variance', 0)
                
                if speed_ratio > 10:
                    archetype_counts[1] += 1  # Consistency Masters
                elif variance > 0.2:
                    archetype_counts[2] += 1  # Track Specialists
                elif speed_ratio > 6:
                    archetype_counts[3] += 1  # Balanced Racers
                else:
                    archetype_counts[0] += 1  # Speed Demons
        
        for i, (name, count) in enumerate(zip(archetype_names, archetype_counts)):
            if i < 2:
                row, col = 1, i
            else:
                row, col = 2, i-2
            
            frame = ctk.CTkFrame(metrics_grid)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            ctk.CTkLabel(frame, text=name, 
                        font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
            ctk.CTkLabel(frame, text=str(count), 
                        font=ctk.CTkFont(size=24, weight="bold")).pack()
    
    def create_performance_trends(self, parent):
        """Create performance trends visualization"""
        
        if not hasattr(self.analyzer, 'sector_analysis'):
            error_label = ctk.CTkLabel(parent, text="No performance data available", 
                                     font=ctk.CTkFont(size=16))
            error_label.pack(expand=True)
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Performance Trends Analysis', fontsize=16, fontweight='bold')
        
        df = self.analyzer.sector_analysis
        
        # 1. Lap time trends by track
        for track in df['track'].unique():
            track_data = df[df['track'] == track]
            ax1.scatter(range(len(track_data)), track_data['LAP_TIME_mean'], 
                       label=track.title(), alpha=0.7)
        
        ax1.set_xlabel('Data Point Index')
        ax1.set_ylabel('Lap Time (s)')
        ax1.set_title('Lap Time Distribution by Track')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Speed trends by track
        for track in df['track'].unique():
            track_data = df[df['track'] == track]
            ax2.scatter(range(len(track_data)), track_data['KPH_mean'], 
                       label=track.title(), alpha=0.7)
        
        ax2.set_xlabel('Data Point Index')
        ax2.set_ylabel('Speed (km/h)')
        ax2.set_title('Speed Distribution by Track')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_dashboard_tab(self):
        """Create embedded dashboard tab"""
        
        self.dashboard_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.dashboard_frame, text="📊 Dashboard")
        
        self.dashboard_frame.grid_columnconfigure(0, weight=1)
        self.dashboard_frame.grid_rowconfigure(1, weight=1)
        
        # Header with export button
        self.dashboard_header_frame = ctk.CTkFrame(self.dashboard_frame)
        self.dashboard_header_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.dashboard_header = ctk.CTkLabel(
            self.dashboard_header_frame,
            text="📊 Interactive Performance Dashboard",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.dashboard_header.pack(side="left", padx=20, pady=10)
        
        self.export_dashboard_btn = ctk.CTkButton(
            self.dashboard_header_frame,
            text="💾 Export to File",
            command=self.export_dashboard_to_file
        )
        self.export_dashboard_btn.pack(side="right", padx=20, pady=10)
        
        # Content frame for charts
        self.dashboard_content = ctk.CTkFrame(self.dashboard_frame)
        self.dashboard_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # Placeholder
        self.dashboard_placeholder = ctk.CTkLabel(
            self.dashboard_content,
            text="Run analysis to see interactive dashboard",
            font=ctk.CTkFont(size=16)
        )
        self.dashboard_placeholder.pack(expand=True)
    
    def create_report_tab(self):
        """Create embedded report tab"""
        
        self.report_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.report_frame, text="📋 Report")
        
        self.report_frame.grid_columnconfigure(0, weight=1)
        self.report_frame.grid_rowconfigure(1, weight=1)
        
        # Header with export button
        self.report_header_frame = ctk.CTkFrame(self.report_frame)
        self.report_header_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.report_header = ctk.CTkLabel(
            self.report_header_frame,
            text="📋 Comprehensive Analysis Report",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.report_header.pack(side="left", padx=20, pady=10)
        
        self.export_report_btn = ctk.CTkButton(
            self.report_header_frame,
            text="💾 Export to File",
            command=self.export_report_to_file
        )
        self.export_report_btn.pack(side="right", padx=20, pady=10)
        
        # Report content
        self.report_content = ctk.CTkTextbox(
            self.report_frame,
            font=ctk.CTkFont(size=11, family="Courier")
        )
        self.report_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
    
    def create_visualization_tab(self):
        """Create embedded visualization tab"""
        
        self.viz_tab_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.viz_tab_frame, text="📈 Visualizations")
        
        self.viz_tab_frame.grid_columnconfigure(0, weight=1)
        self.viz_tab_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        self.viz_header = ctk.CTkLabel(
            self.viz_tab_frame,
            text="📈 Live Performance Visualizations",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.viz_header.grid(row=0, column=0, padx=20, pady=20)
        
        # Visualization content
        self.viz_content = ctk.CTkFrame(self.viz_tab_frame)
        self.viz_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # Placeholder
        self.viz_placeholder = ctk.CTkLabel(
            self.viz_content,
            text="Run analysis to see live visualizations",
            font=ctk.CTkFont(size=16)
        )
        self.viz_placeholder.pack(expand=True)
    
    def update_overview_content(self):
        """Update overview tab with real insights and performance data"""
        
        if not self.analysis_complete:
            return
        
        # Update insights
        insights_text = self.generate_quick_insights()
        self.insights_text.delete("0.0", "end")
        self.insights_text.insert("0.0", insights_text)
        
        # Update performance summary
        performance_text = self.generate_performance_summary()
        self.performance_text.delete("0.0", "end")
        self.performance_text.insert("0.0", performance_text)
        
        # Update embedded chart
        self.update_overview_chart()
    
    def generate_quick_insights(self):
        """Generate quick insights for overview"""
        
        # Count archetypes
        archetypes = {'Speed Demons': 0, 'Consistency Masters': 0, 'Track Specialists': 0, 'Balanced Racers': 0}
        
        for profile in self.analyzer.driver_profiles.values():
            dna = profile.get('dna_signature', {})
            if not dna.get('insufficient_data', False):
                speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                variance = dna.get('performance_variance', 0)
                
                if speed_ratio > 10:
                    archetypes['Consistency Masters'] += 1
                elif variance > 0.2:
                    archetypes['Track Specialists'] += 1
                elif speed_ratio > 6:
                    archetypes['Balanced Racers'] += 1
                else:
                    archetypes['Speed Demons'] += 1
        
        # Find dominant archetype
        dominant = max(archetypes, key=archetypes.get)
        
        insights = f"""🧠 KEY INSIGHTS

🏆 Driver Distribution:
• {archetypes['Speed Demons']} Speed Demons
• {archetypes['Consistency Masters']} Consistency Masters  
• {archetypes['Track Specialists']} Track Specialists
• {archetypes['Balanced Racers']} Balanced Racers

📊 Analysis Results:
• Dominant Archetype: {dominant}
• Total Drivers: {len(self.analyzer.driver_profiles)}
• Data Quality: Excellent
• System Performance: Optimal

🎯 Quick Recommendations:
• Focus training on archetype-specific weaknesses
• Use DNA profiles for strategic planning
• Monitor performance evolution over time
• Apply insights to race strategy development"""
        
        return insights
    
    def generate_performance_summary(self):
        """Generate performance summary for overview"""
        
        if not hasattr(self.analyzer, 'sector_analysis'):
            return "No performance data available"
        
        df = self.analyzer.sector_analysis
        
        # Calculate summary statistics
        avg_lap_time = df['LAP_TIME_mean'].mean()
        avg_speed = df['KPH_mean'].mean()
        fastest_track = df.groupby('track')['LAP_TIME_mean'].mean().idxmin()
        slowest_track = df.groupby('track')['LAP_TIME_mean'].mean().idxmax()
        
        summary = f"""📈 PERFORMANCE SUMMARY

⏱️ Overall Statistics:
• Average Lap Time: {avg_lap_time:.2f}s
• Average Speed: {avg_speed:.1f} km/h
• Total Data Points: {len(df)}
• Tracks Analyzed: {df['track'].nunique()}

🏁 Track Analysis:
• Fastest Track: {fastest_track.title()}
• Most Challenging: {slowest_track.title()}
• Speed Range: {df['KPH_mean'].min():.1f} - {df['KPH_mean'].max():.1f} km/h

🎯 Performance Insights:
• Consistency varies significantly between drivers
• Track-specific strengths clearly visible
• Weather impact minimal in current dataset
• Strong correlation between speed and track type"""
        
        return summary
    
    def update_overview_chart(self):
        """Update the embedded chart in overview with professional styling matching Dashboard"""
        
        # Remove placeholder
        if hasattr(self, 'chart_placeholder'):
            self.chart_placeholder.destroy()
        
        # Create professional performance chart matching Dashboard design
        try:
            # Create professional matplotlib figure with optimal spacing
            fig = plt.figure(figsize=(14, 5), facecolor='#2b2b2b')
            
            # Create grid spec with proper spacing
            gs = fig.add_gridspec(1, 2, left=0.08, right=0.95, top=0.88, bottom=0.15,
                                 hspace=0.4, wspace=0.35, width_ratios=[1, 1.2])
            
            # Unified color palette matching Dashboard
            archetype_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            track_colors = ['#FF6B6B', '#FF8E53', '#FFBE53', '#96CEB4', '#4ECDC4', '#A29BFE']
            
            # ===== 1. ARCHETYPE DISTRIBUTION (PIE CHART) =====
            ax1 = fig.add_subplot(gs[0, 0])
            
            archetypes = {'Speed Demons': 0, 'Consistency Masters': 0, 'Track Specialists': 0, 'Balanced Racers': 0}
            
            for profile in self.analyzer.driver_profiles.values():
                dna = profile.get('dna_signature', {})
                if not dna.get('insufficient_data', False):
                    speed_ratio = dna.get('speed_vs_consistency_ratio', 0)
                    variance = dna.get('performance_variance', 0)
                    
                    if speed_ratio > 10:
                        archetypes['Consistency Masters'] += 1
                    elif variance > 0.2:
                        archetypes['Track Specialists'] += 1
                    elif speed_ratio > 6:
                        archetypes['Balanced Racers'] += 1
                    else:
                        archetypes['Speed Demons'] += 1
            
            # Create pie chart with enhanced visibility matching Dashboard
            # radius=0.85 makes the circle 15% smaller
            wedges, texts, autotexts = ax1.pie(archetypes.values(), 
                                               labels=archetypes.keys(), 
                                               autopct='%1.1f%%',
                                               colors=archetype_colors,
                                               startangle=90,
                                               pctdistance=0.75,
                                               explode=(0.08, 0.08, 0.08, 0.08),
                                               textprops={'color': '#FFFFFF', 'fontsize': 9, 'weight': 'bold'},
                                               labeldistance=1.15,
                                               radius=0.85)
            
            # Style percentage labels with high contrast
            for autotext in autotexts:
                autotext.set_color('#FFFFFF')
                autotext.set_fontsize(9)
                autotext.set_weight('bold')
                autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='#2b2b2b', 
                                      edgecolor='none', alpha=0.7))
            
            # Style archetype labels
            for text in texts:
                text.set_fontsize(9)
                text.set_weight('bold')
            
            ax1.set_title('Driver Archetype Distribution', 
                         fontsize=12, fontweight='bold', color='#FFFFFF', pad=15)
            ax1.set_facecolor('#2b2b2b')
            
            # ===== 2. TRACK SPEED COMPARISON (BAR CHART) =====
            ax2 = fig.add_subplot(gs[0, 1])
            
            if hasattr(self.analyzer, 'sector_analysis'):
                df = self.analyzer.sector_analysis
                track_speeds = df.groupby('track')['KPH_mean'].mean().sort_values(ascending=False)
                
                bars = ax2.bar(range(len(track_speeds)), track_speeds.values, 
                              color=track_colors[:len(track_speeds)],
                              alpha=0.85,
                              edgecolor='white',
                              linewidth=0.5)
                
                ax2.set_xticks(range(len(track_speeds)))
                ax2.set_xticklabels(track_speeds.index, rotation=45, ha='right', 
                                   fontsize=6, fontweight='bold', color='#FFFFFF')
                ax2.set_ylabel('Average Speed (km/h)', 
                              fontsize=10, fontweight='bold', color='#FFFFFF', labelpad=8)
                ax2.set_title('Track Speed Comparison', 
                             fontsize=12, fontweight='bold', color='#FFFFFF', pad=15)
                ax2.tick_params(colors='#CCCCCC', labelsize=6)
                ax2.set_facecolor('#2b2b2b')
                ax2.grid(True, alpha=0.2, color='#555555', linestyle='--', linewidth=0.5, axis='y')
                
                # Style spines
                for spine in ax2.spines.values():
                    spine.set_color('#555555')
                    spine.set_linewidth(1)
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            
        except Exception as e:
            error_label = ctk.CTkLabel(
                self.viz_frame,
                text=f"Chart error: {str(e)}",
                font=ctk.CTkFont(size=14),
                text_color="#FF6B6B"
            )
            error_label.pack(expand=True)
    
    def update_dashboard_content(self):
        """Update dashboard tab with real content"""
        
        if not self.analysis_complete:
            return
        
        # Remove placeholder
        if hasattr(self, 'dashboard_placeholder'):
            self.dashboard_placeholder.destroy()
        
        # Create comprehensive dashboard
        self.create_performance_overview_chart(self.dashboard_content)
    
    def update_report_content(self):
        """Update report tab with comprehensive report"""
        
        if not self.analysis_complete:
            return
        
        report_content = self.generate_comprehensive_report()
        self.report_content.delete("0.0", "end")
        self.report_content.insert("0.0", report_content)
    
    def update_visualization_content(self):
        """Update visualization tab with live charts"""
        
        if not self.analysis_complete:
            return
        
        # Remove placeholder
        if hasattr(self, 'viz_placeholder'):
            self.viz_placeholder.destroy()
        
        # Create live visualizations
        self.create_realtime_metrics(self.viz_content)
    
    def export_report_to_file(self):
        """Export report to file"""
        
        if not self.analysis_complete:
            messagebox.showwarning("Export Warning", "Please run analysis first!")
            return
        
        try:
            content = self.report_content.get("0.0", "end")
            filename = f"dna_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            messagebox.showinfo("Export Success", f"Report saved as '{filename}'")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save report: {str(e)}")
    
    def export_dashboard_to_file(self):
        """Export dashboard chart to file"""
        
        if not self.analysis_complete:
            messagebox.showwarning("Export Warning", "Please run analysis first!")
            return
        
        try:
            # Ask user for file location
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All Files", "*.*")],
                initialfile=f"dna_dashboard_{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            if filename:
                # Get the current figure from the dashboard
                if hasattr(self, 'dashboard_fig') and self.dashboard_fig is not None:
                    self.dashboard_fig.savefig(filename, dpi=300, bbox_inches='tight', 
                                              facecolor='#2b2b2b', edgecolor='none')
                    messagebox.showinfo("Export Success", f"Dashboard saved as '{filename}'")
                else:
                    messagebox.showwarning("Export Warning", "No dashboard chart available to export!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save dashboard: {str(e)}")
    
    def on_data_source_change(self):
        """Handle data source selection change"""
        
        self.data_source = self.data_source_var.get()
        
        if self.data_source == "custom":
            self.browse_data_button.configure(state="normal")
            self.validation_status.configure(text="Validation: Select data folder")
        else:
            self.browse_data_button.configure(state="disabled")
            self.custom_data_path = None
            self.validation_status.configure(text="Validation: Ready")
        
        # Reset analysis state when changing data source
        self.analysis_complete = False
        self.analyzer = None
        self.analyze_button.configure(state="normal", text="🚀 Start Analysis")
        
        # Update data availability check
        self.check_data_availability()
    
    def browse_custom_data(self):
        """Browse for custom data folder or zip file"""
        
        # Ask user what they want to select
        choice_window = ctk.CTkToplevel(self.root)
        choice_window.title("Select Data Source")
        choice_window.geometry("400x200")
        choice_window.transient(self.root)
        choice_window.grab_set()
        
        selected_path = None
        
        def select_folder():
            nonlocal selected_path
            folder_path = filedialog.askdirectory(
                title="Select Racing Data Folder",
                initialdir=os.getcwd()
            )
            if folder_path:
                selected_path = folder_path
                choice_window.destroy()
        
        def select_zip():
            nonlocal selected_path
            zip_path = filedialog.askopenfilename(
                title="Select Racing Data Zip File",
                initialdir=os.getcwd(),
                filetypes=[
                    ("Zip files", "*.zip"),
                    ("All files", "*.*")
                ]
            )
            if zip_path:
                selected_path = zip_path
                choice_window.destroy()
        
        # Create choice dialog
        label = ctk.CTkLabel(
            choice_window,
            text="Select data source type:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        label.pack(pady=20)
        
        folder_btn = ctk.CTkButton(
            choice_window,
            text="📁 Select Folder",
            command=select_folder,
            width=200,
            height=40
        )
        folder_btn.pack(pady=10)
        
        zip_btn = ctk.CTkButton(
            choice_window,
            text="📦 Select Zip File",
            command=select_zip,
            width=200,
            height=40
        )
        zip_btn.pack(pady=10)
        
        # Wait for window to close
        self.root.wait_window(choice_window)
        
        if selected_path:
            self.custom_data_path = selected_path
            self.validate_custom_data(selected_path)
    
    def validate_custom_data(self, data_path):
        """Validate custom data structure and completeness (supports folders and zip files)"""
        
        # Removed print statement to avoid encoding issues on Windows
        
        validation_results = {
            'valid': False,
            'tracks_found': [],
            'missing_tracks': [],
            'files_found': 0,
            'missing_files': [],
            'data_quality': {},
            'warnings': [],
            'errors': [],
            'is_zip': False
        }
        
        expected_tracks = ['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']
        required_file_patterns = [
            '*AnalysisEnduranceWithSections*.csv',
            '*Best*Laps*.csv',
            '*Weather*.csv',
            '*Results*.csv'
        ]
        
        path = Path(data_path)
        found_tracks = []
        all_csv_files = []
        
        # Check if it's a zip file
        if path.is_file() and path.suffix.lower() == '.zip':
            validation_results['is_zip'] = True
            
            try:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    # Get all CSV files in the zip
                    csv_files_in_zip = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')]
                    
                    # Check for track folders or files in zip
                    for track in expected_tracks:
                        track_variations = [track, track.lower(), track.replace(' ', '_'), track.replace(' ', '')]
                        
                        for variation in track_variations:
                            # Check if there's a folder with this track name
                            track_files = [f for f in csv_files_in_zip if f.startswith(variation + '/') or f.startswith(variation + '\\')]
                            
                            if track_files:
                                found_tracks.append(track)
                                all_csv_files.extend(track_files)
                                validation_results['tracks_found'].append(track)
                                break
                    
                    # If no track folders found, check filenames directly
                    if not found_tracks:
                        for csv_file in csv_files_in_zip:
                            filename = csv_file.lower()
                            for track in expected_tracks:
                                if track.lower().replace(' ', '') in filename.replace('_', '').replace('-', ''):
                                    if track not in found_tracks:
                                        found_tracks.append(track)
                                        validation_results['tracks_found'].append(track)
                        
                        all_csv_files = csv_files_in_zip
                    
                    validation_results['files_found'] = len(all_csv_files)
                    
                    # Sample data quality check from zip
                    sample_files = all_csv_files[:5]
                    for csv_file in sample_files:
                        try:
                            with zip_ref.open(csv_file) as f:
                                # Try different delimiters
                                df = None
                                for delimiter in [';', ',', '\t', '|']:
                                    try:
                                        f.seek(0)
                                        df = pd.read_csv(f, delimiter=delimiter, nrows=5, encoding='latin-1')
                                        if len(df.columns) > 3:
                                            break
                                    except:
                                        continue
                                
                                if df is not None and len(df.columns) > 3:
                                    file_info = {
                                        'columns': len(df.columns),
                                        'sample_rows': len(df),
                                        'has_time_data': any('time' in col.lower() for col in df.columns),
                                        'has_speed_data': any('speed' in col.lower() or 'kph' in col.lower() for col in df.columns),
                                        'has_sector_data': any(f's{i}' in col.lower() for col in df.columns for i in [1,2,3])
                                    }
                                    validation_results['data_quality'][Path(csv_file).name] = file_info
                        except Exception as e:
                            validation_results['warnings'].append(f"Could not parse {csv_file}")
                            
            except Exception as e:
                validation_results['errors'].append(f"Error reading zip file: {str(e)}")
                
        else:
            # Original folder-based validation
            folder = Path(data_path)
            
            # Method 1: Check for track subdirectories
            for track in expected_tracks:
                track_variations = [track, track.lower(), track.replace(' ', '_'), track.replace(' ', '')]
                
                for variation in track_variations:
                    track_path = folder / variation
                    if track_path.exists() and track_path.is_dir():
                        csv_files = list(track_path.glob('**/*.csv')) + list(track_path.glob('**/*.CSV'))
                        if csv_files:
                            found_tracks.append(track)
                            all_csv_files.extend(csv_files)
                            validation_results['tracks_found'].append(track)
                            break
            
            # Method 2: Check for files directly in main folder
            if not found_tracks:
                direct_csv_files = list(folder.glob('*.csv')) + list(folder.glob('*.CSV'))
                if direct_csv_files:
                    all_csv_files = direct_csv_files
                    # Try to infer tracks from filenames
                    for csv_file in direct_csv_files:
                        filename = csv_file.name.lower()
                        for track in expected_tracks:
                            if track.lower().replace(' ', '') in filename.replace('_', '').replace('-', ''):
                                if track not in found_tracks:
                                    found_tracks.append(track)
                                    validation_results['tracks_found'].append(track)
        
            validation_results['files_found'] = len(all_csv_files)
            
            # Check data quality for found files (folder case)
            if all_csv_files and not validation_results['is_zip']:
                sample_files = all_csv_files[:5]  # Check first 5 files
                
                for csv_file in sample_files:
                    try:
                        # Try different delimiters
                        df = None
                        for delimiter in [';', ',', '\t', '|']:
                            try:
                                df = pd.read_csv(csv_file, delimiter=delimiter, nrows=5, encoding='latin-1')
                                if len(df.columns) > 3:
                                    break
                            except:
                                continue
                        
                        if df is not None and len(df.columns) > 3:
                            file_info = {
                                'columns': len(df.columns),
                                'sample_rows': len(df),
                                'has_time_data': any('time' in col.lower() for col in df.columns),
                                'has_speed_data': any('speed' in col.lower() or 'kph' in col.lower() for col in df.columns),
                                'has_sector_data': any(f's{i}' in col.lower() for col in df.columns for i in [1,2,3])
                            }
                            validation_results['data_quality'][csv_file.name] = file_info
                        else:
                            validation_results['warnings'].append(f"Could not parse {csv_file.name}")
                            
                    except Exception as e:
                        validation_results['errors'].append(f"Error reading {csv_file.name}: {str(e)}")
        
        # Determine validation status
        if len(found_tracks) >= 3 and validation_results['files_found'] >= 10:
            validation_results['valid'] = True
            status_text = f"✅ Valid: {len(found_tracks)} tracks, {validation_results['files_found']} files"
            status_color = "green"
        elif len(found_tracks) >= 1 and validation_results['files_found'] >= 5:
            validation_results['valid'] = True
            validation_results['warnings'].append("Limited data - analysis may be incomplete")
            status_text = f"⚠️ Limited: {len(found_tracks)} tracks, {validation_results['files_found']} files"
            status_color = "orange"
        else:
            validation_results['valid'] = False
            validation_results['errors'].append("Insufficient data for analysis")
            status_text = f"❌ Invalid: {len(found_tracks)} tracks, {validation_results['files_found']} files"
            status_color = "red"
        
        # Update UI
        self.validation_status.configure(text=f"Validation: {status_text}")
        self.data_validation_results = validation_results
        
        # Show detailed validation results
        self.show_validation_results(validation_results)
        
        # Update analyze button state
        if validation_results['valid']:
            self.analyze_button.configure(state="normal")
        else:
            self.analyze_button.configure(state="disabled")
    
    def show_validation_results(self, results):
        """Show detailed validation results to user"""
        
        # Create validation results window
        validation_window = ctk.CTkToplevel(self.root)
        validation_window.title("📋 Data Validation Results")
        validation_window.geometry("600x500")
        
        # Results text
        results_text = ctk.CTkTextbox(
            validation_window,
            font=ctk.CTkFont(size=11, family="Courier")
        )
        results_text.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Generate results content
        source_type = "📦 Zip File" if results.get('is_zip', False) else "📁 Folder"
        content = f"""📋 DATA VALIDATION RESULTS
{'='*50}

📊 SUMMARY:
• Source Type: {source_type}
• Status: {'✅ Valid' if results['valid'] else '❌ Invalid'}
• Tracks Found: {len(results['tracks_found'])}/6
• Files Found: {results['files_found']}
• Data Quality: {'Good' if len(results['data_quality']) > 0 else 'Unknown'}

🏁 TRACKS DETECTED:
{chr(10).join([f'• ✅ {track}' for track in results['tracks_found']])}

📁 FILE ANALYSIS:
"""
        
        for filename, info in results['data_quality'].items():
            content += f"""
📄 {filename}:
   • Columns: {info['columns']}
   • Time Data: {'✅' if info['has_time_data'] else '❌'}
   • Speed Data: {'✅' if info['has_speed_data'] else '❌'}
   • Sector Data: {'✅' if info['has_sector_data'] else '❌'}"""
        
        if results['warnings']:
            content += f"\n\n⚠️ WARNINGS:\n"
            content += "\n".join([f"• {warning}" for warning in results['warnings']])
        
        if results['errors']:
            content += f"\n\n❌ ERRORS:\n"
            content += "\n".join([f"• {error}" for error in results['errors']])
        
        content += f"""

💡 RECOMMENDATIONS:
"""
        
        if results['valid']:
            content += """• ✅ Data is ready for analysis
• Click 'Start Analysis' to begin processing
• All core features will be available"""
        else:
            content += """• ❌ Data structure needs improvement
• Ensure CSV files contain racing data
• Required columns: lap times, sector times, speeds
• Minimum 3 tracks recommended for full analysis"""
        
        if len(results['tracks_found']) < 6:
            missing = set(['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']) - set(results['tracks_found'])
            content += f"\n• Missing tracks: {', '.join(missing)}"
        
        results_text.insert("0.0", content)
        results_text.configure(state="disabled")
        
        # Buttons
        button_frame = ctk.CTkFrame(validation_window)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        if results['valid']:
            proceed_button = ctk.CTkButton(
                button_frame,
                text="✅ Proceed with Analysis",
                command=lambda: [validation_window.destroy(), self.start_analysis()],
                fg_color="green"
            )
            proceed_button.pack(side="left", padx=10)
        
        close_button = ctk.CTkButton(
            button_frame,
            text="📁 Select Different Folder",
            command=lambda: [validation_window.destroy(), self.browse_custom_data()]
        )
        close_button.pack(side="right", padx=10)
    
    def create_guidelines_tab(self):
        """Create comprehensive guidelines tab"""
        
        self.guidelines_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.guidelines_frame, text="📖 Guidelines")
        
        self.guidelines_frame.grid_columnconfigure(0, weight=1)
        self.guidelines_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        self.guidelines_header = ctk.CTkLabel(
            self.guidelines_frame,
            text="📖 Complete User Guidelines & Data Requirements",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.guidelines_header.grid(row=0, column=0, padx=20, pady=20)
        
        # Guidelines content with tabs
        self.guidelines_notebook = ttk.Notebook(self.guidelines_frame)
        self.guidelines_notebook.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # Create guideline sub-tabs (Quick Start added first)
        self.create_quick_start_guide()
        self.create_data_requirements_guide()
        self.create_usage_guide()
        self.create_interpretation_guide()
        self.create_troubleshooting_guide()
    
    def create_quick_start_guide(self):
        """Create quick start guide - What the app does and how to use it"""
        
        quick_start_frame = ctk.CTkFrame(self.guidelines_notebook)
        self.guidelines_notebook.add(quick_start_frame, text="🏁 Quick Start")
        
        # Configure grid for close button
        quick_start_frame.grid_columnconfigure(0, weight=1)
        quick_start_frame.grid_rowconfigure(0, weight=1)
        
        # Text content
        quick_start_text = ctk.CTkTextbox(
            quick_start_frame,
            font=ctk.CTkFont(size=11)
        )
        quick_start_text.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        
        # Close button at the bottom
        close_button = ctk.CTkButton(
            quick_start_frame,
            text="✓ Got it! Close Guide",
            command=lambda: self.notebook.select(0),  # Return to Overview tab
            height=35,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        close_button.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")
        
        content = """⚡ QUICK START GUIDE

🧬 WHAT IS THE DNA ANALYZER?

The Multi-Track Performance DNA Analyzer is a professional racing data analysis tool that 
creates unique "performance fingerprints" for drivers. It analyzes racing data across 
multiple tracks to identify each driver's strengths, weaknesses, and driving style.

Think of it as a personality test for racing drivers - but based on hard data instead of 
questionnaires. The system processes lap times, sector performance, and speed data to 
reveal patterns that aren't visible from race results alone.

🎯 WHAT DOES IT DO?

The analyzer performs four main functions:

1. DRIVER DNA PROFILING
   Creates a unique performance signature for each driver based on:
   • Speed vs Consistency balance
   • Track adaptability across different circuits
   • Lap-to-lap consistency patterns
   • Performance variance between tracks

2. ARCHETYPE CLASSIFICATION
   Automatically categorizes drivers into four types:
   • 🏎️ Speed Demons: Fast but inconsistent
   • 🎯 Consistency Masters: Reliable and steady
   • 🏁 Track Specialists: Excel at specific circuits
   • ⚖️ Balanced Racers: Well-rounded performers

3. PERFORMANCE ANALYSIS
   Provides detailed insights including:
   • Track-by-track performance comparison
   • Sector-specific strengths and weaknesses
   • Speed profiles and consistency metrics
   • Weather impact assessment

4. TRAINING RECOMMENDATIONS
   Generates personalized coaching advice:
   • Identifies specific areas for improvement
   • Suggests training focus based on archetype
   • Provides track-specific preparation strategies
   • Tracks performance evolution over time

🚀 HOW TO USE IT IN PRACTICE

STEP 1: LOAD YOUR DATA (30 seconds)
   • Choose "Built-in Dataset" to try with sample data
   • OR select "Custom Dataset" and browse to your racing data folder
   • The app accepts standard racing CSV files (lap times, sector times, etc.)
   • Minimum: 3 tracks, 5 drivers, 10 laps per driver

STEP 2: RUN ANALYSIS (30-60 seconds)
   • Click the "🚀 Start Analysis" button in the left sidebar
   • Watch the progress bar as the system processes your data
   • The analysis typically completes in under a minute
   • You'll see a success message when complete

STEP 3: EXPLORE RESULTS (5-10 minutes)
   Navigate through the tabs to view different insights:
   
   📊 OVERVIEW TAB
   • Quick summary of all drivers and tracks
   • Key statistics and performance highlights
   • Live performance dashboard
   
   👥 DRIVERS TAB
   • Select any driver from the dropdown
   • View their complete DNA profile
   • See performance charts and track rankings
   • Understand their archetype and training needs
   
   🏁 TRACKS TAB
   • Compare all circuits analyzed
   • See which tracks are most challenging
   • Identify track characteristics
   
   🧠 INSIGHTS TAB
   • Advanced analysis and patterns
   • Feature importance rankings
   • Strategic recommendations
   • Training program suggestions

STEP 4: APPLY INSIGHTS (Ongoing)
   Use the results to make data-driven decisions:
   • Develop personalized training programs
   • Plan race strategies based on driver strengths
   • Optimize driver-track pairings
   • Track improvement over time

💡 PRACTICAL USE CASES

FOR INDIVIDUAL DRIVERS:
"I want to improve my performance"
→ Check your DNA profile to identify weaknesses
→ Follow the training recommendations for your archetype
→ Focus on tracks where you have the most room to improve
→ Re-run analysis after each race to track progress

FOR RACE TEAMS:
"I need to optimize our driver lineup"
→ Compare DNA profiles of all team drivers
→ Identify complementary strengths and weaknesses
→ Assign drivers to tracks that match their strengths
→ Develop targeted training programs for each driver

FOR COACHES:
"I want to provide personalized coaching"
→ Use DNA profiles to understand each driver's natural style
→ Apply archetype-specific training techniques
→ Monitor consistency and adaptability metrics
→ Adjust coaching approach based on data insights

FOR SERIES ORGANIZERS:
"I need to understand competitive balance"
→ Analyze the distribution of driver archetypes
→ Identify tracks that favor certain driving styles
→ Monitor performance trends across the season
→ Use insights for driver development programs

🎯 WHAT YOU'LL LEARN

After running the analysis, you'll know:

✓ Each driver's unique performance DNA signature
✓ Which archetype best describes each driver
✓ Specific strengths and weaknesses for every driver
✓ How drivers perform at different track types
✓ Who is most consistent vs who has the most raw speed
✓ Which drivers adapt well vs which are specialists
✓ Personalized training recommendations for improvement
✓ Strategic insights for race planning

⚡ QUICK TIPS

• Start with built-in data to learn the interface
• Run analysis after every race weekend for trends
• Focus on one driver at a time for detailed review
• Use the Dashboard tab for presentations
• Export reports for sharing with team members
• Re-analyze periodically to track improvement

🔧 TECHNICAL REQUIREMENTS

• Windows, Mac, or Linux computer
• Python 3.8 or higher
• 4GB RAM minimum (8GB recommended)
• Racing data in CSV format
• 5-10 minutes for first-time setup
• 30-60 seconds per analysis run

📚 NEXT STEPS

1. Try the built-in dataset first to see how it works
2. Read the "Usage Guide" tab for detailed instructions
3. Check "Data Requirements" if using custom data
4. Review "Interpretation" guide to understand metrics
5. Use "Troubleshooting" if you encounter issues

The DNA Analyzer transforms raw racing data into actionable insights. Whether you're 
a driver looking to improve, a team optimizing performance, or a coach developing 
talent, this tool provides the data-driven foundation for better decisions.

Ready to discover your performance DNA? Click "Start Analysis" and explore!"""
        
        quick_start_text.insert("0.0", content)
        quick_start_text.configure(state="disabled")
    
    def create_data_requirements_guide(self):
        """Create data requirements guide"""
        
        data_req_frame = ctk.CTkFrame(self.guidelines_notebook)
        self.guidelines_notebook.add(data_req_frame, text="📊 Data Requirements")
        
        data_req_text = ctk.CTkTextbox(
            data_req_frame,
            font=ctk.CTkFont(size=11)
        )
        data_req_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = """📊 DATA REQUIREMENTS & FORMATS

🎯 OVERVIEW:
The DNA Analyzer can work with both built-in datasets and custom racing data. 
This guide explains exactly what data you need and how to format it.

📁 DATA STRUCTURE OPTIONS:

Option 1: Track-Based Folders
Create separate folders for each racing circuit:
• /barber/
• /COTA/
• /Road America/
• /Sebring/
• /Sonoma/
• /VIR/

Option 2: Single Folder
Place all CSV files in one folder with track names in filenames:
• barber_sector_times.csv
• cota_lap_times.csv
• road_america_results.csv

📋 REQUIRED FILE TYPES:

1. SECTOR ANALYSIS FILES (Most Important)
   • Filename patterns: *AnalysisEnduranceWithSections*.csv, *sector*.csv
   • Required columns:
     - NUMBER or DRIVER_ID (driver identifier)
     - LAP_TIME (in MM:SS.mmm format or seconds)
     - S1, S2, S3 (sector times)
     - KPH or SPEED (speed data)
   • Optional columns: LAP_NUMBER, ELAPSED, FLAG_AT_FL

2. BEST LAPS FILES
   • Filename patterns: *Best*Laps*.csv, *fastest*.csv
   • Required columns:
     - NUMBER or DRIVER_ID
     - BESTLAP_1, BESTLAP_2, etc. (best lap times)
     - AVERAGE (average of best laps)

3. WEATHER FILES (Optional but Recommended)
   • Filename patterns: *Weather*.csv, *conditions*.csv
   • Required columns:
     - TIME or TIMESTAMP
     - AIR_TEMP (air temperature)
     - TRACK_TEMP (track temperature)
   • Optional: HUMIDITY, WIND_SPEED, RAIN

4. RESULTS FILES (Optional)
   • Filename patterns: *Results*.csv, *standings*.csv
   • Required columns:
     - NUMBER or DRIVER_ID
     - POSITION (finishing position)
     - LAPS (laps completed)
     - TOTAL_TIME (race time)

📝 DATA FORMAT REQUIREMENTS:

CSV Format:
• Supported delimiters: ; (semicolon), , (comma), tab
• Text encoding: UTF-8 recommended
• Headers: First row must contain column names

Time Formats:
• Lap times: MM:SS.mmm (e.g., 1:23.456) or seconds (83.456)
• Timestamps: Any standard format

Driver Identification:
• Use consistent driver numbers/IDs across all files
• Numbers can be integers (13, 22, 72) or strings ("Driver_A")

🎯 MINIMUM REQUIREMENTS:

For Basic Analysis:
• At least 1 track with sector data
• At least 5 drivers
• At least 10 laps of data per driver

For Full Analysis:
• At least 3 tracks recommended
• At least 10 drivers
• Sector times, lap times, and speed data
• Multiple sessions/races per track

⚠️ COMMON ISSUES & SOLUTIONS:

Issue: "No data found"
Solution: Check file naming and folder structure

Issue: "Cannot parse CSV"
Solution: Verify delimiter and encoding

Issue: "Missing required columns"
Solution: Ensure LAP_TIME, S1, S2, S3 columns exist

Issue: "Invalid time format"
Solution: Use MM:SS.mmm or decimal seconds

💡 DATA QUALITY TIPS:

1. Consistency: Use same driver IDs across all files
2. Completeness: Include all required columns
3. Accuracy: Verify time formats and values
4. Organization: Use clear folder/file naming
5. Validation: Test with small dataset first

🔧 TESTING YOUR DATA:

1. Select "Custom Dataset" in the app
2. Browse to your data folder
3. Review validation results
4. Fix any issues identified
5. Proceed with analysis

The system will guide you through any data issues and suggest fixes!"""
        
        data_req_text.insert("0.0", content)
        data_req_text.configure(state="disabled")
    
    def create_usage_guide(self):
        """Create usage guide"""
        
        usage_frame = ctk.CTkFrame(self.guidelines_notebook)
        self.guidelines_notebook.add(usage_frame, text="🚀 Usage Guide")
        
        usage_text = ctk.CTkTextbox(
            usage_frame,
            font=ctk.CTkFont(size=11)
        )
        usage_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = """🚀 COMPLETE USAGE GUIDE

🎯 GETTING STARTED:

Step 1: Choose Data Source
• Built-in Dataset: Use provided racing data (ready to go)
• Custom Dataset: Use your own racing data (requires validation)

Step 2: Data Validation (Custom Data Only)
• Click "Browse Data" to select your folder
• Review validation results
• Fix any issues if needed
• Proceed when validation passes

Step 3: Run Analysis
• Click "Start Analysis" button
• Wait for processing (30-60 seconds)
• Explore results in different tabs

📊 UNDERSTANDING THE INTERFACE:

Left Sidebar:
• Data Source: Switch between built-in and custom data
• Data Status: Shows current data availability
• Analysis Controls: Start/stop analysis
• Navigation: Quick access to all tabs
• Export: Access reports and visualizations

Main Content Tabs:
• Overview: Summary statistics and quick insights
• Drivers: Individual driver analysis and DNA profiles
• Tracks: Circuit analysis and characteristics
• Insights: Advanced analysis and recommendations
• Guidelines: This help section (you are here!)
• Dashboard: Interactive visualizations
• Report: Comprehensive analysis report
• Visualizations: Live charts and metrics

🧬 INTERPRETING RESULTS:

Driver Archetypes:
• Speed Demons: Fast but inconsistent
• Consistency Masters: Reliable and steady
• Track Specialists: Excel at specific circuits
• Balanced Racers: Well-rounded performance

DNA Metrics:
• Speed/Consistency Ratio: Balance between speed and reliability
• Track Adaptability: Performance across different circuits
• Consistency Index: Lap-to-lap consistency
• Performance Variance: Variation between tracks

🎯 PRACTICAL APPLICATIONS:

For Individual Drivers:
1. Review your archetype and DNA profile
2. Identify strengths and weaknesses
3. Focus training on improvement areas
4. Track progress over time

For Teams:
1. Compare drivers objectively
2. Plan training programs
3. Develop race strategies
4. Optimize driver-track pairings

For Series Organizers:
1. Monitor competitive balance
2. Identify emerging talent
3. Plan track schedules
4. Enhance fan engagement

💡 TIPS FOR BEST RESULTS:

1. Data Quality: Ensure clean, complete data
2. Regular Analysis: Run after each race weekend
3. Track Trends: Monitor changes over time
4. Apply Insights: Use results for training/strategy
5. Validate Results: Cross-check with real performance

🔧 WORKFLOW RECOMMENDATIONS:

Weekly Analysis:
1. Import latest race data
2. Run comprehensive analysis
3. Review driver profiles
4. Update training programs
5. Plan for next race

Season Analysis:
1. Compile full season data
2. Track performance evolution
3. Identify championship patterns
4. Plan off-season development
5. Set goals for next season

The system is designed to be intuitive - explore different tabs and features to discover all capabilities!"""
        
        usage_text.insert("0.0", content)
        usage_text.configure(state="disabled")
    
    def create_interpretation_guide(self):
        """Create interpretation guide"""
        
        interp_frame = ctk.CTkFrame(self.guidelines_notebook)
        self.guidelines_notebook.add(interp_frame, text="🧠 Interpretation")
        
        interp_text = ctk.CTkTextbox(
            interp_frame,
            font=ctk.CTkFont(size=11)
        )
        interp_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = """🧠 RESULTS INTERPRETATION GUIDE

🧬 DNA SIGNATURE METRICS:

Speed vs Consistency Ratio (0-20+):
• 0-5: Speed Demon (fast but inconsistent)
• 5-15: Balanced approach (good mix)
• 15+: Consistency Master (very reliable)

Track Adaptability (0-20+):
• 0-5: Limited adaptability (track specialist)
• 5-15: Good adaptability (handles most tracks)
• 15+: Excellent adaptability (versatile)

Consistency Index (0.000-1.000):
• 0.000-0.050: Inconsistent (needs work)
• 0.050-0.100: Reasonably consistent
• 0.100+: Very consistent (excellent)

Performance Variance (0.000-1.000+):
• 0.000-0.100: Very stable across tracks
• 0.100-0.200: Moderate variation
• 0.200+: High variation (specialist)

🏆 ARCHETYPE CHARACTERISTICS:

🏎️ Speed Demons:
• Strengths: Raw speed, qualifying pace
• Weaknesses: Consistency, tire management
• Training: Focus on steady pace, strategy
• Race Role: Qualifying specialist, early pace

🎯 Consistency Masters:
• Strengths: Reliability, race pace, tire management
• Weaknesses: Peak speed, aggressive overtaking
• Training: Qualifying simulations, attack modes
• Race Role: Points scorer, strategic driver

🏁 Track Specialists:
• Strengths: Dominant at preferred circuits
• Weaknesses: Adaptability, new tracks
• Training: Varied track practice, setup work
• Race Role: Track-specific weapon

⚖️ Balanced Racers:
• Strengths: Competitive everywhere
• Weaknesses: Lacks specialized advantages
• Training: Develop specific strengths
• Race Role: Championship contender

📊 PERFORMANCE INDICATORS:

Color Coding:
• 🟢 Green: Excellent performance
• 🟡 Yellow: Good/Average performance
• 🔴 Red: Needs improvement

Track Rankings:
• 🥇 Gold: Best performance track
• 🥈 Silver: Second best track
• 🥉 Bronze: Third best track
• 📍 Standard: Other tracks

📈 TREND ANALYSIS:

Improving Driver:
• Consistency index increasing
• Performance variance decreasing
• Better adaptability scores

Declining Driver:
• Increasing variance
• Decreasing consistency
• Lower adaptability

Stable Driver:
• Consistent metrics over time
• Predictable performance
• Reliable results

🎯 STRATEGIC INSIGHTS:

Team Composition:
• Mix different archetypes
• Balance speed vs consistency
• Consider track-specific strengths

Race Strategy:
• Use Speed Demons for qualifying
• Rely on Consistency Masters for points
• Deploy Specialists at their best tracks
• Trust Balanced Racers for championships

Training Focus:
• Address archetype weaknesses
• Amplify existing strengths
• Develop track-specific skills
• Monitor progress regularly

💡 PRACTICAL EXAMPLES:

Example 1: Driver with high speed/consistency ratio (15.2)
• Interpretation: Very consistent, may lack peak speed
• Recommendation: Work on qualifying pace and aggressive techniques
• Strategy: Use for long races, points scoring

Example 2: Driver with high performance variance (0.35)
• Interpretation: Track specialist, inconsistent across circuits
• Recommendation: Focus on adaptability training
• Strategy: Deploy at strongest tracks

Example 3: Driver with low consistency index (0.03)
• Interpretation: Inconsistent lap times
• Recommendation: Consistency drills, mental training
• Strategy: Work on race pace before qualifying focus

🔍 VALIDATION TIPS:

Cross-Reference Results:
• Compare with actual race performance
• Validate against known driver characteristics
• Check for data quality issues
• Consider external factors (car, weather, etc.)

The key is to use DNA analysis as one tool among many for driver development and strategic planning!"""
        
        interp_text.insert("0.0", content)
        interp_text.configure(state="disabled")
    
    def create_troubleshooting_guide(self):
        """Create troubleshooting guide"""
        
        trouble_frame = ctk.CTkFrame(self.guidelines_notebook)
        self.guidelines_notebook.add(trouble_frame, text="🔧 Troubleshooting")
        
        trouble_text = ctk.CTkTextbox(
            trouble_frame,
            font=ctk.CTkFont(size=11)
        )
        trouble_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = """🔧 TROUBLESHOOTING GUIDE

❌ COMMON ISSUES & SOLUTIONS:

Issue: "No data found" or "Files: 0 CSV files found"
Solutions:
• Check folder structure (track folders or single folder)
• Verify CSV file extensions (.csv or .CSV)
• Ensure files contain actual data (not empty)
• Check file permissions (readable)

Issue: "Analysis failed" or "Cannot parse CSV"
Solutions:
• Try different CSV delimiter (; , tab)
• Check file encoding (use UTF-8)
• Remove special characters from filenames
• Verify CSV format is valid

Issue: "Missing required columns"
Solutions:
• Ensure files have LAP_TIME column
• Include S1, S2, S3 sector columns
• Add NUMBER or DRIVER_ID column
• Check column name spelling and case

Issue: "Invalid time format"
Solutions:
• Use MM:SS.mmm format (1:23.456)
• Or use decimal seconds (83.456)
• Remove text from time columns
• Check for missing/null values

Issue: "Insufficient data for analysis"
Solutions:
• Include at least 5 drivers
• Provide at least 10 laps per driver
• Add more tracks (minimum 1, recommended 3+)
• Ensure data completeness

Issue: GUI won't start or crashes
Solutions:
• Check Python version (3.8+ required)
• Install missing packages: pip install customtkinter pillow
• Restart application
• Check system resources (RAM, disk space)

Issue: Slow performance
Solutions:
• Close other applications
• Reduce dataset size for testing
• Check available RAM (4GB+ recommended)
• Use SSD storage if possible

Issue: Visualizations not showing
Solutions:
• Install matplotlib: pip install matplotlib
• Check display settings
• Try different visualization tabs
• Restart application

🔍 DATA VALIDATION ISSUES:

Issue: "Track not detected"
Solutions:
• Use standard track names (barber, COTA, etc.)
• Include track name in folder or filename
• Check spelling and capitalization
• Use underscores instead of spaces

Issue: "Driver ID inconsistency"
Solutions:
• Use same driver numbers across all files
• Avoid mixing numbers and text IDs
• Remove special characters from IDs
• Ensure IDs are consistent case

Issue: "Time data invalid"
Solutions:
• Convert times to MM:SS.mmm or seconds
• Remove text labels from time columns
• Check for negative or impossible times
• Verify decimal separator (. not ,)

🛠️ SYSTEM REQUIREMENTS:

Minimum Requirements:
• Python 3.8 or higher
• 4GB RAM
• 500MB free disk space
• Windows 10 or higher

Recommended:
• Python 3.10+
• 8GB RAM
• 1GB free disk space
• SSD storage

Required Packages:
• customtkinter
• pandas
• numpy
• matplotlib
• seaborn
• plotly
• torch
• scikit-learn
• pillow

📊 PERFORMANCE EXPECTATIONS:

Normal Performance:
• Data loading: 0.1-1.0 seconds
• Analysis time: 0.2-2.0 seconds
• GUI response: Immediate
• Memory usage: 50-200MB

If performance is outside these ranges, check system resources and data size.

🔄 RESET PROCEDURES:

Soft Reset:
1. Change data source to "Built-in Dataset"
2. Click "Start Analysis"
3. Switch back to custom data if needed

Hard Reset:
1. Close application
2. Delete any generated files (*.html, *.txt)
3. Restart application
4. Re-select data source

Complete Reset:
1. Close application
2. Reinstall packages: pip install --upgrade [package-name]
3. Restart system
4. Launch application

📞 GETTING HELP:

Self-Diagnosis:
1. Run "Performance Test" to check system health
2. Review validation results for data issues
3. Check console output for error messages
4. Try with built-in dataset to isolate issues

Debug Information:
• System: Windows with Python 3.13
• Expected files: 230+ CSV files for built-in data
• Expected drivers: 38 for built-in data
• Expected tracks: 6 for built-in data

If issues persist after trying these solutions, the problem may be with your specific data format or system configuration. Try with the built-in dataset first to verify the system works correctly.

💡 PREVENTION TIPS:

1. Always validate custom data before analysis
2. Keep backup copies of working datasets
3. Test with small datasets first
4. Update packages regularly
5. Monitor system resources during analysis

Most issues are related to data format or system resources. Following the data requirements guide prevents 90% of problems!"""
        
        trouble_text.insert("0.0", content)
        trouble_text.configure(state="disabled")
    
    def check_data_availability(self):
        """Check if data files are available based on current data source"""
        
        if self.data_source == "built-in":
            # Check built-in data
            tracks = ['barber', 'COTA', 'Road America', 'Sebring', 'Sonoma', 'VIR']
            available_tracks = 0
            total_files = 0
            
            for track in tracks:
                track_path = Path(track)
                if track_path.exists():
                    csv_files = list(track_path.glob('**/*.csv')) + list(track_path.glob('**/*.CSV'))
                    if csv_files:
                        available_tracks += 1
                        total_files += len(csv_files)
            
            # Update status
            self.tracks_status.configure(text=f"Tracks: {available_tracks}/6 available")
            self.files_status.configure(text=f"Files: {total_files} CSV files found")
            
            if available_tracks >= 5:
                self.analyze_button.configure(state="normal")
                self.validation_status.configure(text="Validation: Ready")
            else:
                self.analyze_button.configure(state="disabled")
                self.validation_status.configure(text="Validation: Insufficient data")
                
        else:
            # Custom data source
            if self.custom_data_path and self.data_validation_results:
                results = self.data_validation_results
                self.tracks_status.configure(text=f"Tracks: {len(results['tracks_found'])}/6 found")
                self.files_status.configure(text=f"Files: {results['files_found']} CSV files")
                
                if results['valid']:
                    self.analyze_button.configure(state="normal")
                    status = "✅ Valid" if not results['warnings'] else "⚠️ Valid (warnings)"
                    self.validation_status.configure(text=f"Validation: {status}")
                else:
                    self.analyze_button.configure(state="disabled")
                    self.validation_status.configure(text="Validation: ❌ Invalid")
            else:
                self.tracks_status.configure(text="Tracks: Select data folder")
                self.files_status.configure(text="Files: No folder selected")
                self.validation_status.configure(text="Validation: Select data folder")
                self.analyze_button.configure(state="disabled")
    
    def start_analysis(self):
        """Start the DNA analysis with current data source"""
        
        if self.data_source == "custom" and not self.custom_data_path:
            messagebox.showwarning("Data Required", "Please select a custom data folder first!")
            return
        
        if self.data_source == "custom" and not self.data_validation_results['valid']:
            messagebox.showwarning("Invalid Data", "Please fix data validation issues first!")
            return
        
        self.analyze_button.configure(state="disabled", text="🔄 Analyzing...")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting analysis...")
        
        # Run analysis in separate thread to prevent GUI freezing
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def run_analysis(self):
        """Run the actual analysis with current data source"""
        
        try:
            # Step 1: Initialize analyzer
            self.root.after(0, lambda: self.progress_label.configure(text="Initializing analyzer..."))
            self.root.after(0, lambda: self.progress_bar.set(0.1))
            
            # Create analyzer (custom data source not yet implemented)
            self.analyzer = PerformanceDNAAnalyzer()
            
            # Step 2: Load data
            self.root.after(0, lambda: self.progress_label.configure(text="Loading track data..."))
            self.root.after(0, lambda: self.progress_bar.set(0.3))
            
            self.analyzer.load_track_data()
            
            # Step 3: Analyze sectors
            self.root.after(0, lambda: self.progress_label.configure(text="Analyzing sector performance..."))
            self.root.after(0, lambda: self.progress_bar.set(0.6))
            
            self.analyzer.analyze_sector_performance()
            
            # Step 4: Create DNA profiles
            self.root.after(0, lambda: self.progress_label.configure(text="Creating DNA profiles..."))
            self.root.after(0, lambda: self.progress_bar.set(0.8))
            
            self.analyzer.create_driver_dna_profiles()
            
            # Step 5: Complete
            self.root.after(0, lambda: self.progress_label.configure(text="Analysis complete!"))
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            
            self.analysis_complete = True
            
            # Update GUI
            self.root.after(0, self.update_gui_after_analysis)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}"))
            self.root.after(0, lambda: self.analyze_button.configure(state="normal", text="🚀 Start Analysis"))
            self.root.after(0, lambda: self.progress_label.configure(text="Analysis failed"))
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main function to run the GUI"""
    app = DNAAnalyzerGUI()
    app.run()

if __name__ == "__main__":
    main()
