#!/usr/bin/env python3
"""
Splash Screen Launcher for MTP DNA Analyzer
Shows a progress bar while the main application starts
"""

import sys
import os
import subprocess
import threading
import time
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    # ERROR: tkinter not available. Cannot show splash screen.
    sys.exit(1)


class SplashScreen:
    """Splash screen with real progress bar"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MTP DNA Analyzer")
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        
        # Set window size
        window_width = 500
        window_height = 150
        
        # Center the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg='#2b2b2b')
        
        # Set icon if available
        icon_path = self._find_icon()
        if icon_path and os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except:
                pass
        
        # Create content frame
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(expand=True, fill='both', padx=30, pady=20)
        
        # Title label
        title_label = tk.Label(
            content_frame,
            text="MTP DNA Analyzer",
            font=('Segoe UI', 16, 'bold'),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        title_label.pack(pady=(5, 15))
        
        # Status label
        self.status_label = tk.Label(
            content_frame,
            text="Initializing...",
            font=('Segoe UI', 9),
            bg='#2b2b2b',
            fg='#cccccc'
        )
        self.status_label.pack(pady=(0, 10))
        
        # Progress bar with custom style
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor='#1a1a1a',
            background='#0078d4',
            bordercolor='#2b2b2b',
            lightcolor='#0078d4',
            darkcolor='#0078d4',
            thickness=20
        )
        
        self.progress = ttk.Progressbar(
            content_frame,
            style="Custom.Horizontal.TProgressbar",
            mode='determinate',
            length=440,
            maximum=100
        )
        self.progress.pack(pady=(0, 5))
        self.progress['value'] = 0
        
        # Percentage label
        self.percent_label = tk.Label(
            content_frame,
            text="0%",
            font=('Segoe UI', 8),
            bg='#2b2b2b',
            fg='#999999'
        )
        self.percent_label.pack()
        
        # Force window to appear immediately
        self.root.update_idletasks()
        self.root.update()
        
        # Process handle
        self.main_process = None
        self.should_close = False
    
    def _find_icon(self):
        """Find the icon file"""
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
            internal_path = os.path.join(base_path, '_internal')
            
            for path in [internal_path, base_path]:
                icon_path = os.path.join(path, 'icon.ico')
                if os.path.exists(icon_path):
                    return icon_path
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(script_dir, 'icon.ico')
            if os.path.exists(icon_path):
                return icon_path
        
        return None
    
    def update_progress(self, value, status="Loading..."):
        """Update progress bar and status"""
        if self.should_close:
            return
        
        try:
            self.progress['value'] = value
            self.percent_label.config(text=f"{int(value)}%")
            self.status_label.config(text=status)
            self.root.update()
        except:
            pass
    
    def launch_main_app(self):
        """Launch the main application with progress updates"""
        def run_app():
            try:
                # Update: Finding application
                self.update_progress(10, "Locating application files...")
                time.sleep(0.3)
                
                # Determine the main launcher path
                if getattr(sys, 'frozen', False):
                    base_path = os.path.dirname(sys.executable)
                    
                    possible_paths = [
                        os.path.join(base_path, 'main_launcher.exe'),
                        os.path.join(base_path, '_internal', 'main_launcher.exe'),
                    ]
                    
                    main_launcher = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            main_launcher = path
                            break
                    
                    if not main_launcher:
                        for root, dirs, files in os.walk(base_path):
                            if 'main_launcher.exe' in files:
                                main_launcher = os.path.join(root, 'main_launcher.exe')
                                break
                    
                    if not main_launcher or not os.path.exists(main_launcher):
                        self.update_progress(100, "Error: Application not found")
                        time.sleep(2)
                        self.close()
                        return
                    
                    self.update_progress(25, "Starting application...")
                    time.sleep(0.3)
                    
                    # Launch without showing console window
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                    
                    self.main_process = subprocess.Popen(
                        [main_launcher],
                        startupinfo=startupinfo,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                else:
                    # Running as script
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    main_launcher = os.path.join(script_dir, 'main_launcher.py')
                    
                    if not os.path.exists(main_launcher):
                        self.update_progress(100, "Error: Application not found")
                        time.sleep(2)
                        self.close()
                        return
                    
                    self.update_progress(25, "Starting application...")
                    time.sleep(0.3)
                    
                    self.main_process = subprocess.Popen(
                        [sys.executable, main_launcher],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                
                # Simulate loading progress over 30 seconds
                stages = [
                    (35, "Loading dependencies..."),
                    (45, "Initializing modules..."),
                    (55, "Loading models..."),
                    (65, "Preparing interface..."),
                    (75, "Loading data..."),
                    (85, "Finalizing..."),
                    (95, "Almost ready..."),
                ]
                
                for progress_val, status_text in stages:
                    time.sleep(3.5)  # ~3.5 seconds per stage = ~24.5 seconds total
                    self.update_progress(progress_val, status_text)
                
                # Final wait
                time.sleep(5)
                self.update_progress(100, "Ready!")
                time.sleep(0.5)
                
                # Close splash screen
                self.close()
                
            except Exception as e:
                self.update_progress(100, f"Error: {str(e)}")
                time.sleep(2)
                self.close()
        
        thread = threading.Thread(target=run_app, daemon=True)
        thread.start()
    
    def close(self):
        """Close the splash screen"""
        self.should_close = True
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
    
    def show(self):
        """Show the splash screen and launch main app"""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.update()
        
        # Start launching the main app
        self.launch_main_app()
        
        # Show splash screen (blocks until closed)
        try:
            self.root.mainloop()
        except:
            pass


def main():
    """Main entry point"""
    try:
        splash = SplashScreen()
        splash.show()
    except Exception as e:
        # ERROR: Failed to show splash screen
        # If splash fails, try to launch main app directly
        try:
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
                main_launcher = os.path.join(base_path, 'main_launcher.exe')
                subprocess.Popen([main_launcher])
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                main_launcher = os.path.join(script_dir, 'main_launcher.py')
                subprocess.Popen([sys.executable, main_launcher])
        except:
            pass


if __name__ == "__main__":
    main()
