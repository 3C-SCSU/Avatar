# Versioning Guidelines

This document outlines the versioning process and guidelines for the Avatar BCI project.

## Overview

The Avatar BCI project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and maintains a detailed [changelog](CHANGELOG.md) using the [Keep a Changelog](https://keepachangelog.com/) format.

**🤖 AUTOMATED VERSIONING IS NOW ACTIVE!**

This project uses automated versioning based on [Conventional Commits](https://www.conventionalcommits.org/). Version numbers and changelog entries are generated automatically from your commit messages.

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

## 🚀 Automated Release Process

### How Releases Work Now
**✅ Fully Automated! No manual steps required.**

1. **Make Changes**: Use conventional commits for your work
2. **Create PR**: Push to feature branch, create pull request  
3. **Merge to Main**: PR gets merged to main branch
4. **Automation Triggers**: GitHub Actions runs automatically
5. **Release Created**: Version bumped, changelog updated, tag created

### Conventional Commit Examples
```bash
# Minor version bump (new features)
git commit -m "feat(bci): add support for 32-channel headsets"
git commit -m "feat(drone): add autonomous flight mode"

# Patch version bump (bug fixes)
git commit -m "fix(gui): resolve brainwave chart rendering"
git commit -m "fix(nao): fix robot connection timeout"

# Major version bump (breaking changes)  
git commit -m "feat(api)!: redesign prediction interface"
# or
git commit -m "feat(api): redesign prediction interface

BREAKING CHANGE: The prediction API now returns different format"
```

### Manual Testing (Optional)
Test automation locally before pushing:
```bash
python scripts/auto_version.py
```

### Override Automation
Add `[skip ci]` to commit message to skip automation:
```bash
git commit -m "docs: fix typo [skip ci]"
```

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

## 🤖 Automated Versioning System

### Current Implementation
✅ **Active Features:**
- Automatic version bumping based on conventional commits
- Automated changelog generation from commit history  
- GitHub Actions workflow for releases
- Git tagging and GitHub releases
- Semantic versioning enforcement

### How It Works
1. **Commit Analysis**: `scripts/auto_version.py` analyzes commits since last release
2. **Version Determination**: Determines bump type based on conventional commit types
3. **File Updates**: Updates `VERSION` and `CHANGELOG.md` automatically
4. **Release Creation**: GitHub Actions creates tags and releases

### Commit Type → Version Mapping
| Commit Type | Version Bump | Example |
|-------------|--------------|---------|
| `feat` | minor (1.1.0) | `feat(bci): add new headset support` |
| `fix` | patch (1.0.1) | `fix(drone): resolve timeout issue` |
| `feat!` or `BREAKING CHANGE` | major (2.0.0) | `feat(api)!: change interface` |
| `docs`, `chore`, etc. | patch (1.0.1) | `docs: update README` |

### GitHub Actions Workflow
Located in `.github/workflows/release.yml`:
- Triggers on push to `main` branch
- Runs `scripts/auto_version.py`
- Commits version updates with `[skip ci]`
- Creates git tags and GitHub releases

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
