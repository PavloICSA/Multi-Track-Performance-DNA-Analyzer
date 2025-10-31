@echo off
REM Build script for MTP DNA Analyzer Installer
REM This script automates the entire build and packaging process

echo ========================================
echo MTP DNA Analyzer - Installer Builder
echo ========================================
echo.

REM Step 1: Check if PyInstaller is installed
echo [1/5] Checking PyInstaller installation...
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)
echo PyInstaller is ready.
echo.

REM Step 2: Clean previous builds
echo [2/5] Cleaning previous builds...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "installer_output" rmdir /s /q "installer_output"
echo Clean complete.
echo.

REM Step 3: Build executable with PyInstaller
echo [3/5] Building executable with PyInstaller...
echo This may take several minutes...
pyinstaller build_spec.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    pause
    exit /b 1
)
echo Executable built successfully.
echo.

REM Step 4: Check if Inno Setup is installed
echo [4/5] Checking Inno Setup installation...
set "INNO_PATH=F:\Inno Setup 6\ISCC.exe"
if not exist "%INNO_PATH%" (
    echo.
    echo WARNING: Inno Setup not found at default location.
    echo Please install Inno Setup 6 from: https://jrsoftware.org/isdl.php
    echo.
    echo After installation, run this script again or manually compile:
    echo   "%INNO_PATH%" installer_script.iss
    echo.
    echo The application executable is ready in: dist\MTP DNA Analyzer\
    pause
    exit /b 0
)
echo Inno Setup found.
echo.

REM Step 5: Create installer with Inno Setup
echo [5/5] Creating installer with Inno Setup...
"%INNO_PATH%" installer_script.iss
if errorlevel 1 (
    echo ERROR: Inno Setup compilation failed
    pause
    exit /b 1
)
echo.

echo ========================================
echo BUILD COMPLETE!
echo ========================================
echo.
echo Installer created: installer_output\MTP_DNA_Analyzer_Setup.exe
echo Application folder: dist\MTP DNA Analyzer\
echo.
echo You can now distribute the setup.exe file to users.
echo.
pause
