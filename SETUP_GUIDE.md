# Avatar Project Setup Guide

**Version: v1.0.0** | [Changelog](CHANGELOG.md)

## Overview
This guide helps you set up and run the Avatar BCI project correctly with all dependencies and proper paths.

## Fixed Issues

### 1. Path Issues Fixed
- **BCI Client Import Path**: Fixed the import path from `'random-forest-prediction'` to `'prediction-random-forest/tensorflow'` in `GUI5.py` (line 29)
- **All QML Files**: Verified all QML files exist in the correct locations
- **Python Module Paths**: Verified all Python module imports for file-shuffler, file-unify-labels, and file-remove8channel

### 2. Dependencies Installed
- `matplotlib` - Required by the `Developers/devCharts.py` module
- `requests` - Required by the BCI connection module
- `scikit-learn` - Required for machine learning models

## Prerequisites

### Hardware
- [OpenBCI Headset](https://shop.openbci.com/products/ultracortex-mark-iv) with [Cyton 16 Channel board](https://shop.openbci.com/products/cyton-daisy-biosensing-boards-16-channel)
- [DJI Tello Edu Drone](https://www.amazon.com/DJI-CP-TL-00000026-02-Tello-EDU/dp/B07TZG2NNT)
- NAO Robot (optional)

### Software
- Python 3.13 or higher
- macOS (tested), Linux, or Windows

## Installation

### 1. Clone or Fork the Repository
```bash
git clone https://github.com/3C-SCSU/Avatar.git
cd Avatar
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install matplotlib requests scikit-learn
```

## Running the Application

### Option 1: Use the Run Script (Recommended)
```bash
./run_gui.sh
```

### Option 2: Manual Activation
```bash
source venv/bin/activate
python3 GUI5.py
```

## Project Structure

```
Avatar/
├── GUI5.py                          # Main application entry point
├── main.qml                         # Main QML UI file
├── BrainwaveReading.qml            # Brainwave reading interface
├── ManualDroneControl.qml          # Drone control interface
├── ManualNaoControl.qml            # NAO robot control interface
├── BrainwaveVisualization.qml      # Data visualization
├── FileShuffler.qml                # File shuffling utility
├── TransferData.qml                # Data transfer interface
├── Developers.qml                  # Developer statistics
├── prediction-random-forest/       # ML prediction models
│   └── tensorflow/
│       └── client/
│           └── brainflow1.py       # BCI connection handler
├── file-shuffler/                  # File shuffling utilities
│   └── run_file_shuffler.py
├── file-unify-labels/              # Label unification
│   └── unifyTXT.py
├── file-remove8channel/            # 8-channel data removal
│   └── remove8channel.py
├── NAO6/                           # NAO robot integration
│   └── nao_connection.py
├── Developers/                     # Developer tools
│   └── devCharts.py                # Contribution charts
└── GUI5_ManualDroneControl/       # Drone control module
    └── cameraview/
        └── camera_controller.py

```

## Troubleshooting

### Import Errors
If you encounter module import errors:
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install matplotlib requests scikit-learn
```

### QML Loading Errors
Ensure all QML files are in the project root directory:
- main.qml
- BrainwaveReading.qml
- ManualDroneControl.qml
- ManualNaoControl.qml
- BrainwaveVisualization.qml
- FileShuffler.qml
- TransferData.qml
- Developers.qml

### Path Issues
The application expects to be run from the project root directory where `GUI5.py` is located.

### BCI Connection Issues
If BCI connection fails, it will fall back to simulation mode. Check:
- OpenBCI headset is connected
- Serial port is correct (default: `/dev/cu.usbserial-D200PMA1`)
- Drivers are installed

### NAO Robot Connection Issues
If NAO connection fails:
- Ensure `nao_service.py` is running on port 5000
- Check NAO IP address in `NAO6/nao_connection.py`
- Verify network connectivity to the NAO robot

## Features

### Brainwave Reading
- Real-time brainwave data collection from OpenBCI headset
- Machine learning prediction (Random Forest, Deep Learning)
- PyTorch and TensorFlow framework support
- Synthetic data mode for testing

### Drone Control
- Manual control interface
- BCI-driven autopilot
- Live camera stream
- Flight logging

### NAO Robot Control
- Manual animation control
- Sit/stand commands
- Speech and movement choreography

### Developer Tools
- Git contribution statistics
- Commit tier visualization (Gold/Silver/Bronze)
- Ticket tracking by developer

## Development

### Running in Development Mode
```bash
source venv/bin/activate
python3 GUI5.py
```

### Adding Dependencies
```bash
source venv/bin/activate
pip install <package_name>
pip freeze > requirements.txt
```

## Contributing
See [README.md](README.md) for contribution guidelines.

## License
MIT License - See [LICENSE](LICENSE) for details.

## Versioning

### Version Information
- **Current Version**: v1.0.0 (see [VERSION](VERSION) file)
- **Versioning Scheme**: [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md) for detailed version history

### Understanding Version Numbers
- **MAJOR**: Incompatible API changes or major feature overhauls
- **MINOR**: New functionality added in a backwards compatible manner
- **PATCH**: Backwards compatible bug fixes and small improvements

### Checking Your Version
```bash
cat VERSION
```

### Version Compatibility
- BCI Hardware: OpenBCI headsets with Cyton boards
- Python: 3.13 or higher required
- Dependencies: See [requirements.txt](requirements.txt) for exact versions

## Support
- Discord: https://discord.gg/mxbQ7HpPjq
- YouTube: https://www.youtube.com/@3CSCSU
- Email: 3c.scsu@gmail.com

