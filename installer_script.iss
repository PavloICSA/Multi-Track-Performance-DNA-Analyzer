; Inno Setup Script for MTP DNA Analyzer
; This script creates a professional Windows installer

#define MyAppName "MTP DNA Analyzer"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "Performance DNA Team"
#define MyAppExeName "MTP DNA Analyzer.exe"
#define MyAppIcon "icon.ico"

[Setup]
; Application Information
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppCopyright=Copyright (C) 2024 {#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}

; Installation Directories
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output Configuration
OutputDir=installer_output
OutputBaseFilename=MTP_DNA_Analyzer_Setup
SetupIconFile={#MyAppIcon}

; Compression
Compression=lzma2/max
SolidCompression=yes

; UI Configuration
WizardStyle=modern

; Privileges
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Architecture
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main executable and all files from PyInstaller output
Source: "dist\MTP DNA Analyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Additional files
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop shortcut (if selected)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
; Option to launch the application after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Any post-installation tasks can be added here
  end;
end;
