@echo off
echo ========================================
echo Creating GitHub Release v1.0.0
echo ========================================
echo.

REM Check if GitHub CLI is installed
where gh >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: GitHub CLI is not installed!
    echo.
    echo Please install GitHub CLI from: https://cli.github.com/
    echo Or create the release manually using the web interface.
    echo See HOW_TO_CREATE_RELEASE.md for instructions.
    echo.
    pause
    exit /b 1
)

echo Checking authentication...
gh auth status
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Please authenticate with GitHub first:
    echo   gh auth login
    echo.
    pause
    exit /b 1
)

echo.
echo Creating release v1.0.0...
echo.

gh release create v1.0.0 ^
  --title "Multi-Track Performance DNA Analyzer v1.0.0" ^
  --notes-file RELEASE_NOTES.md ^
  "installer_output\MTP_DNA_Analyzer_Setup.exe"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Release created successfully!
    echo ========================================
    echo.
    echo View your release at:
    echo https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases
    echo.
    echo Direct download link:
    echo https://github.com/PavloICSA/Multi-Track-Performance-DNA-Analyzer/releases/download/v1.0.0/MTP_DNA_Analyzer_Setup.exe
    echo.
) else (
    echo.
    echo ERROR: Failed to create release!
    echo Please check the error message above or create the release manually.
    echo See HOW_TO_CREATE_RELEASE.md for instructions.
    echo.
)

pause
