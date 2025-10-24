# Changelog

All notable changes to the Avatar BCI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]







## [1.0.6] - 2025-10-24

### Fixed
- Auto adds the changes to the release description(changelog) (e5d5b0a)

## [1.0.5] - 2025-10-24

### Changed
- Update readme.md (2d2c676)
- Fix (#371): restored functional drone camera view with linux-compatible path handling (363dfb9)

### Fixed
- Release token from github token to release token pat(ci/cd) (79ce9e8)

## [1.0.4] - 2025-10-24

### Changed
- Remove summary (d121bcf)

## [1.0.3] - 2025-10-24

### Changed
- Refactor is_connected property and signals (b619946)
- Enable/disable is_connected boolean flag based on connection status (3aa6b64)
- Add connection status handling for tello drone (5bdf4d5)
- Add ip/port to ui #395 (10967dd)
- Fix (#370): replaced windows-style path formatting with pathlib.path.as_uri() for full cross-platform compatibility (8ae6553)
- Fix (#370, #371): replaced windows-style path formatting with pathlib.path.as_uri() for full cross-platform compatibility (417aea1)

## [1.0.2] - 2025-10-23

### Changed
- Push workflow on pr (8be06c3)

## [1.0.1] - 2025-10-23

### Changed
- Release automation(versioning) (01fe4ae)
- Versioning (b940516)
- Format and sort (7aaa615)
- Format files (9e87fe9)
- Working ci (47cf405)
- Faster ci (d73d988)
- Ci (c5f61c3)
- Ci (1c2cc35)
- Py version (477382d)
- Ignore ds_store (03134a2)
- Create python-app.yml (83cbd04)
- Delete .qtcreator directory (3b1eb24)
- Delete .idea directory (efd1902)
- Delete .ds_store (2c23cd2)
- Fix (f8c5c78)
- Qt creator (7ef1198)
- Git ignore (2b2e70b)
- Fixing paths (dae7130)
- Fixgui5 (9fd55ad)
- Fixuideveloperstab (85373b9)
- Revert "fix/drone camera view 371 v2" (5e3342b)
- Revert "fix/drone flightlog 370 v2" (6eb7714)
- Replaced old jax model files and updated notebooks (9916ef9)
- Revert "add jax random forest for nao control" (90d2e25)
- Enhance run.sh with error handling and python detection (349e05d)
- Correct main block execution flow (bd79133)
- Ux fix (#418): updated window title to 'avatar - bci', moved all tabs to top, and added active tab highlighting (green background, yellow text) (3430879)
- Fix nao control panel ui (#419): resolved log overlap and unified button icon sizing (7a54f20)
- Drone cameraview integration and windows-safe frame path (#371) (807344c)
- Gui v5: drone flight log — backend bridge + qml model; sim fallback + shorter timeouts (fixes #370) (62aa2a5)
- Added new jax deep learning files and cleaned up old versions (f970ced)

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
