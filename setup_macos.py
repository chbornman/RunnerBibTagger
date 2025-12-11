"""
py2app setup script for Bib Tagger macOS app.

Usage:
    python setup_macos.py py2app

This creates a standalone .app bundle in the dist/ folder.
"""

from setuptools import setup

APP = ['bib_tagger_gui.py']
DATA_FILES = [
    ('models', ['models/bib_detector.onnx']),
]
OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'resources/icon.icns',  # Optional - create if you have one
    'plist': {
        'CFBundleName': 'Bib Tagger',
        'CFBundleDisplayName': 'Bib Tagger',
        'CFBundleIdentifier': 'com.bibtagger.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
    },
    'packages': [
        'cv2',
        'numpy',
        'onnxruntime',
        'rapidocr_onnxruntime',
    ],
    'includes': [
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.ttk',
    ],
    'excludes': [
        'torch',
        'torchvision',
        'ultralytics',
        'paddlepaddle',
        'paddleocr',
        'tensorflow',
        'keras',
    ],
}

setup(
    app=APP,
    name='Bib Tagger',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
