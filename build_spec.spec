# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Build main_launcher.exe first
main_launcher = Analysis(
    ['main_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('icon.ico', '.'),
        ('config.py', '.'),
        ('dna_analyzer_gui.py', '.'),
        ('dna_dashboard.py', '.'),
        ('dna_exceptions.py', '.'),
        ('dna_explainability.py', '.'),
        ('dna_feature_engineering.py', '.'),
        ('dna_insights_generator.py', '.'),
        ('dna_logging.py', '.'),
        ('dna_model_inference.py', '.'),
        ('dna_model_registry.py', '.'),
        ('dna_model_trainer.py', '.'),
        ('evaluate_model.py', '.'),
        ('performance_dna_analyzer.py', '.'),
        ('promote_model.py', '.'),
        ('retrain_dna_model.py', '.'),
        ('train_dna_model.py', '.'),
        ('README.md', '.'),
        ('barber', 'barber'),
        ('COTA', 'COTA'),
        ('Road America', 'Road America'),
        ('Sebring', 'Sebring'),
        ('Sonoma', 'Sonoma'),
        ('VIR', 'VIR'),
        ('models', 'models'),
        ('training_artifacts', 'training_artifacts'),
    ],
    hiddenimports=[
        'customtkinter',
        'tkinter',
        'PIL',
        'PIL._tkinter_finder',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'torch',
        'sklearn',
        'scipy',
        'sklearn.ensemble',
        'sklearn.tree',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

main_pyz = PYZ(main_launcher.pure, main_launcher.zipped_data, cipher=block_cipher)

main_exe = EXE(
    main_pyz,
    main_launcher.scripts,
    [],
    exclude_binaries=True,
    name='main_launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

# Build splash_launcher.exe (the main entry point)
splash_launcher = Analysis(
    ['splash_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('icon.ico', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

splash_pyz = PYZ(splash_launcher.pure, splash_launcher.zipped_data, cipher=block_cipher)

splash_exe = EXE(
    splash_pyz,
    splash_launcher.scripts,
    [],
    exclude_binaries=True,
    name='MTP DNA Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    splash_exe,
    splash_launcher.binaries,
    splash_launcher.zipfiles,
    splash_launcher.datas,
    main_exe,
    main_launcher.binaries,
    main_launcher.zipfiles,
    main_launcher.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MTP DNA Analyzer',
)


