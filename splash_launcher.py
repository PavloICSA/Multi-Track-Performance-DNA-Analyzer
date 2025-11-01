#!/usr/bin/env python3
"""
Splash Screen Launcher for MTP DNA Analyzer
Shows a loading animation while the main application starts
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
    print("ERROR: tkinter not available. Cannot show splash screen.")
    sys.exit(1)


class SplashScreen:
    """Simple splash screen with loading animation"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MTP DNA Analyzer")
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Set window size
        window_width = 400
        window_height = 200
        
        # Center the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg='#1a1a1a')
        
        # Set icon if available
        icon_path = self._find_icon()
        if icon_path and os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except:
                pass
        
        # Create content frame
        content_frame = tk.Frame(self.root, bg='#1a1a1a')
        content_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title label
        title_label = tk.Label(
            content_frame,
            text="MTP DNA Analyzer",
            font=('Arial', 18, 'bold'),
            bg='#1a1a1a',
            fg='#ffffff'
        )
        title_label.pack(pady=(10, 5))
        
        # Subtitle label
        subtitle_label = tk.Label(
            content_frame,
            text="Performance Analysis Tool",
            font=('Arial', 10),
            bg='#1a1a1a',
            fg='#cccccc'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Loading label
        self.loading_label = tk.Label(
            content_frame,
            text="Loading",
            font=('Arial', 11),
            bg='#1a1a1a',
            fg='#ffffff'
        )
        self.loading_label.pack(pady=(10, 10))
        
        # Progress bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Splash.Horizontal.TProgressbar",
            troughcolor='#2a2a2a',
            background='#4a9eff',
            bordercolor='#1a1a1a',
            lightcolor='#4a9eff',
            darkcolor='#4a9eff'
        )
        
        self.progress = ttk.Progressbar(
            content_frame,
            style="Splash.Horizontal.TProgressbar",
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=(0, 20))
        self.progress.start(10)
        
        # Animation state
        self.dots = 0
        self.animate_loading()
        
        # Process handle
        self.main_process = None
        self.should_close = False
    
    def _find_icon(self):
        """Find the icon file"""
        if getattr(sys, 'frozen', False):
            # Running as executable
            base_path = os.path.dirname(sys.executable)
            internal_path = os.path.join(base_path, '_internal')
            
            # Check both locations
            for path in [internal_path, base_path]:
                icon_path = os.path.join(path, 'icon.ico')
                if os.path.exists(icon_path):
                    return icon_path
        else:
            # Running as script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(script_dir, 'icon.ico')
            if os.path.exists(icon_path):
                return icon_path
        
        return None
    
    def animate_loading(self):
        """Animate the loading text"""
        if self.should_close:
            return
        
        self.dots = (self.dots + 1) % 4
        dots_text = '.' * self.dots
        self.loading_label.config(text=f"Loading{dots_text}")
        
        self.root.after(500, self.animate_loading)
    
    def launch_main_app(self):
        """Launch the main application in a separate thread"""
        def run_app():
            try:
                # Determine the main launcher path
                if getattr(sys, 'frozen', False):
                    # Running as executable - launch main_launcher.exe
                    base_path = os.path.dirname(sys.executable)
                    main_launcher = os.path.join(base_path, 'main_launcher.exe')
                    
                    if not os.path.exists(main_launcher):
                        print(f"ERROR: main_launcher.exe not found at {main_launcher}")
                        self.close()
                        return
                    
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
                    # Running as script - launch main_launcher.py
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    main_launcher = os.path.join(script_dir, 'main_launcher.py')
                    
                    if not os.path.exists(main_launcher):
                        print(f"ERROR: main_launcher.py not found at {main_launcher}")
                        self.close()
                        return
                    
                    self.main_process = subprocess.Popen(
                        [sys.executable, main_launcher],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                
                # Wait a bit for the main window to appear
                # Adjust this timeout based on your app's startup time
                time.sleep(35)  # 35 seconds to ensure main app is fully loaded
                
                # Close splash screen
                self.close()
                
            except Exception as e:
                print(f"ERROR launching main application: {e}")
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
        # Start launching the main app
        self.launch_main_app()
        
        # Show splash screen (blocks until closed)
        self.root.mainloop()


def main():
    """Main entry point"""
    try:
        splash = SplashScreen()
        splash.show()
    except Exception as e:
        print(f"ERROR: Failed to show splash screen: {e}")
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
