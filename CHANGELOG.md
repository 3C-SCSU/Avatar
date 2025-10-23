# Changelog

All notable changes to the Avatar BCI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2024-10-23

### Added
- Initial stable release of Avatar BCI project
- Brain-Computer Interface (BCI) integration with OpenBCI headset
- Machine learning prediction models (Random Forest, Deep Learning)
- Support for PyTorch, TensorFlow, and JAX frameworks
- Real-time brainwave reading and visualization
- DJI Tello drone control interface with camera stream
- NAO robot control with manual animations and choreography
- Qt5/PySide6 GUI application with multiple control tabs
- File shuffling utilities for privacy-preserving data handling
- Developer statistics and contribution tracking
- Kubernetes deployment configurations
- CI/CD pipeline with GitHub Actions
- Python application workflow automation
- Docker containerization for server components
- Apache Spark integration with Delta Lake
- Secure file transfer capabilities with SFTP
- Data processing utilities for BCI signal analysis
- Multiple GUI interfaces: Brainwave Reading, Manual Controls, Visualization

### Changed
- Code formatting and organization improvements
- Updated Python dependencies to latest stable versions
- Improved project structure and file organization
- Enhanced documentation with setup guides and troubleshooting

### Fixed
- GUI5 path import issues resolved
- UI developer tab functionality corrections
- Qt Creator configuration cleanup
- Serial port connection stability for BCI headset
- File path resolution for cross-platform compatibility

### Removed
- Cleaned up .DS_Store files from repository
- Removed .idea and .qtcreator IDE configuration directories
- Legacy configuration files that caused conflicts

### Security
- Implemented secure file transfer protocols
- Added privacy protection for brainwave data collection
- Ensured IRB compliance for human subject data handling
