# Avatar BCI - Quick Setup Guide

## Requirements
- Python 3.11 or 3.12 (Python 3.14 is NOT compatible)
- macOS with ARM64 (Apple Silicon) or x86_64 (Intel)

## Installation

1. **Create virtual environment** (first time only):
```bash
cd /Users/zeyini/Desktop/Foundations_SE/Avatar_Zeyini
python3.11 -m venv venv
```

2. **Activate virtual environment**:
```bash
source venv/bin/activate
```

3. **Install dependencies** (first time only):
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org PySide6==6.7.3 brainflow djitellopy pandas opencv-python pdf2image paramiko pysftp matplotlib
```

## Running the Application

### Method 1: Using the startup script
```bash
./start_gui.sh
```

### Method 2: Manual startup
```bash
cd /Users/zeyini/Desktop/Foundations_SE/Avatar_Zeyini
source venv/bin/activate
python GUI5.py
```

## Features
- **Brainwave Reading**: Connect to OpenBCI headset and read brainwave data
- **Manual Drone Control**: Control DJI Tello drone manually
- **Manual NAO Control**: Control NAO robot
- **Artificial Intelligence**: ML model training and prediction
- **File Shuffler**: Data preprocessing utilities
- **Transfer Data**: SFTP file transfer
- **Developers**: Team statistics and visualizations

## Troubleshooting

### GUI doesn't open
- Make sure you're using Python 3.11, not 3.14
- Run from native Terminal, not IDE terminal
- Check for permission prompts in System Settings > Privacy & Security

### Import errors
- Make sure virtual environment is activated (`source venv/bin/activate`)
- Reinstall dependencies if needed

### Python version issues
Check your Python version:
```bash
python --version
```
Should show Python 3.11.x

## Notes
- The virtual environment keeps all packages isolated from your system Python
- Your Mac's system software is NOT affected by these installations
- Always activate the venv before running the application



