# Versioning Guidelines

This document outlines the versioning process and guidelines for the Avatar BCI project.

## Overview

The Avatar BCI project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and maintains a detailed [changelog](CHANGELOG.md) using the [Keep a Changelog](https://keepachangelog.com/) format.

## Version Number Format

**Format**: `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)

### When to Increment

#### MAJOR Version (X.y.z)
Increment when you make **incompatible API changes** or **major breaking changes**:
- Changes to BCI interface that break compatibility with existing headsets
- Major GUI overhaul that changes user workflow
- Breaking changes to NAO/drone control APIs
- Database schema changes that require migration
- Python version requirement changes (e.g., dropping Python 3.12 support)

#### MINOR Version (x.Y.z)
Increment when you add **functionality in a backwards compatible manner**:
- New BCI prediction models or frameworks
- Additional drone/robot control features
- New GUI tabs or visualization tools
- New file processing utilities
- Enhanced machine learning capabilities
- New hardware support (additional drone models, etc.)

#### PATCH Version (x.y.Z)
Increment when you make **backwards compatible bug fixes**:
- Bug fixes in BCI data processing
- UI bug fixes and minor improvements
- Documentation updates
- Dependency version updates (security patches)
- Performance optimizations
- Code formatting and cleanup

## File-Based Version Management

### VERSION File
- Location: `./VERSION`
- Contains only the version number (e.g., `1.0.0`)
- Single source of truth for current version
- Updated manually with each release

### Reading Version in Code
```python
def get_version():
    with open('VERSION', 'r') as f:
        return f.read().strip()
```

## Release Process

### 1. Prepare Release
1. **Update VERSION file** with new version number
2. **Update CHANGELOG.md**:
   - Move items from `[Unreleased]` to new version section
   - Add release date
   - Create new empty `[Unreleased]` section
3. **Update version references** in documentation if needed

### 2. Commit and Tag
```bash
# Commit version changes
git add VERSION CHANGELOG.md
git commit -m "Release v1.2.3"

# Create version tag
git tag -a v1.2.3 -m "Release v1.2.3"

# Push changes and tags
git push origin main
git push origin v1.2.3
```

### 3. Update Documentation
- Ensure README.md version badge is updated
- Update any version-specific documentation
- Update hardware compatibility notes if needed

## Changelog Management

### Format
Follow [Keep a Changelog](https://keepachangelog.com/) format with sections:
- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes

### Guidelines
- Write entries from the user's perspective
- Be specific about what changed
- Include relevant PR/issue numbers when applicable
- Group related changes together
- Use present tense ("Add feature" not "Added feature")

### Example Entry
```markdown
## [1.2.0] - 2024-10-23

### Added
- Support for DJI Mini 3 Pro drone control (#123)
- Real-time EEG signal filtering options (#125)
- Export functionality for brainwave data (#127)

### Changed
- Improved GUI responsiveness during BCI data collection (#124)
- Updated TensorFlow models for better prediction accuracy (#126)

### Fixed
- Resolved NAO robot connection timeout issues (#122)
- Fixed camera stream lag in drone control interface (#128)
```

## Pre-Release Versions

For development and testing purposes, use pre-release identifiers:
- **Alpha**: `1.2.0-alpha.1` (early development, unstable)
- **Beta**: `1.2.0-beta.1` (feature complete, testing phase)
- **Release Candidate**: `1.2.0-rc.1` (final testing before release)

## Branch-Specific Versioning

### Main Branch
- Contains stable releases only
- All releases are tagged from main branch
- VERSION file always reflects latest stable version

### Development Branches  
- Use branch names like `feature/new-bci-model` or `fix/drone-connection`
- Don't update VERSION file in feature branches
- Merge to main triggers version update

### Changelog and Versioning Branch
- Special branch for versioning system setup
- Can be merged to main once versioning system is established

## Integration with CI/CD

### Automated Checks
Consider adding automated checks for:
- Version number format validation
- Changelog entry requirements for PRs
- Version consistency across files

### Build Automation
Future enhancements could include:
- Automatic version bumping based on commit messages
- Automated changelog generation from commit history
- Version-specific build artifacts

## Hardware Compatibility Matrix

Maintain compatibility information in changelog for:

### BCI Hardware
- OpenBCI Ultracortex Mark IV
- Cyton 16-channel boards
- Compatible EEG electrodes

### Robots/Drones
- DJI Tello/Tello EDU
- NAO Robot (specific versions)
- Future hardware additions

## Best Practices

### Do's
- ✅ Update VERSION file with every release
- ✅ Maintain detailed changelog entries
- ✅ Test thoroughly before version increments
- ✅ Document breaking changes clearly
- ✅ Follow semantic versioning strictly

### Don'ts
- ❌ Skip version updates for "small" changes
- ❌ Make breaking changes in PATCH versions
- ❌ Forget to update changelog
- ❌ Use inconsistent version numbers across files
- ❌ Release without proper testing

## Version History

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## Questions?

For questions about versioning:
- Check existing [GitHub Issues](https://github.com/3C-SCSU/Avatar/issues)
- Join our [Discord](https://discord.gg/mxbQ7HpPjq)
- Email: 3c.scsu@gmail.com
