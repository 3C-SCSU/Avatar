# Contributing to Avatar BCI Project

Welcome to the Avatar BCI project! This guide will help you contribute effectively using our automated versioning and changelog system.

## 🤖 Automated Versioning

This project uses **automated versioning** based on conventional commits. Your commit messages determine how the version number is bumped and what appears in the changelog.

## 📝 Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. All commit messages must follow this format:

```
<type>(<scope>): <description>

<body>

<footer>
```

### Quick Examples
```bash
feat(bci): add support for 32-channel OpenBCI boards
fix(drone): resolve connection timeout in DJI Tello
docs: update installation guide
refactor(gui): improve brainwave visualization performance
```

## 🏷️ Commit Types

| Type | Version Bump | Changelog Section | Description |
|------|--------------|-------------------|-------------|
| `feat` | **minor** | Added | New features |
| `fix` | **patch** | Fixed | Bug fixes |
| `perf` | **patch** | Changed | Performance improvements |
| `docs` | **patch** | Changed | Documentation changes |
| `style` | **patch** | Changed | Code style/formatting |
| `refactor` | **patch** | Changed | Code refactoring |
| `test` | **patch** | Changed | Test additions/changes |
| `chore` | **patch** | Changed | Maintenance tasks |
| `ci` | **patch** | Changed | CI/CD changes |
| `build` | **patch** | Changed | Build system changes |

### 💥 Breaking Changes
Add `!` after the type or include `BREAKING CHANGE:` in the footer for **major** version bumps:

```bash
feat(ml)!: change prediction model API
fix(bci): resolve headset detection

BREAKING CHANGE: The BCI interface now requires initialization
```

## 🎯 Scopes (Optional)

Use scopes to indicate the affected module:

- `bci` - Brain-Computer Interface components
- `drone` - DJI Tello drone control
- `nao` - NAO robot integration
- `gui` - Qt/PySide GUI application
- `ml` - Machine learning models
- `server` - Server-side components
- `docs` - Documentation
- `ci` - Continuous Integration
- `k8s` - Kubernetes deployments

## 🔄 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/new-bci-model
# or
git checkout -b fix/drone-connection-timeout
```

### 2. Make Changes with Conventional Commits
```bash
# Add new feature
git commit -m "feat(bci): add EEG signal filtering options"

# Fix bug
git commit -m "fix(drone): resolve camera stream lag"

# Update documentation
git commit -m "docs: add troubleshooting for NAO robot setup"
```

### 3. Create Pull Request
- Push your branch to GitHub
- Create a Pull Request to `main`
- The automation will handle versioning when merged

### 4. Automated Release Process
When your PR is merged to `main`:
1. 🤖 GitHub Actions analyzes your commits
2. 📈 Determines version bump (major/minor/patch)
3. 📝 Updates `VERSION` file
4. 📋 Updates `CHANGELOG.md`
5. 🏷️ Creates git tag
6. 🚀 Creates GitHub release

## 🛠️ Setting Up Commit Template

Configure git to use our commit template:

```bash
git config commit.template .gitcommit-template
```

Now when you run `git commit`, you'll see helpful prompts.

## 📋 Manual Version Testing

Test the automation locally:

```bash
python scripts/auto_version.py
```

This will show you what version bump would occur based on recent commits.

## 🎨 Commit Message Best Practices

### ✅ Good Examples
```bash
feat(bci): add support for Cyton 32-channel boards
fix(drone): resolve takeoff command timeout after 10 seconds
docs: update README with new hardware requirements
perf(ml): optimize TensorFlow model inference by 40%
refactor(gui): extract brainwave chart into separate component
test(bci): add unit tests for signal processing pipeline
```

### ❌ Bad Examples
```bash
# Too vague
fix: bug fix

# Not imperative mood
feat: added new feature

# Missing scope when helpful
update gui

# Too long subject line
feat(bci): add comprehensive support for all OpenBCI hardware variants including the new 32-channel boards with improved signal processing
```

## 🔍 Changelog Preview

Your commits automatically generate changelog entries:

```markdown
## [1.2.0] - 2024-10-23

### Added
- Add support for Cyton 32-channel boards (bci) (a1b2c3d)
- Add EEG signal filtering options (bci) (d4e5f6g)

### Fixed
- Resolve takeoff command timeout after 10 seconds (drone) (g7h8i9j)
- Fix camera stream lag issues (drone) (j1k2l3m)

### Changed
- Optimize TensorFlow model inference by 40% (ml) (m4n5o6p)
- Extract brainwave chart into separate component (gui) (p7q8r9s)
```

## 🚀 Release Types

- **Patch (1.0.X)**: Bug fixes, documentation, small improvements
- **Minor (1.X.0)**: New features, backwards-compatible changes
- **Major (X.0.0)**: Breaking changes, major API modifications

## 🤝 University Contributions

### For approved bounty submissions:
1. Create branch: `feature-bounty` (e.g., `file_shuffler-bounty`)
2. Use conventional commits
3. Open PR with bounty label
4. Update project wiki after merge

### External Contributions
1. Fork the repository
2. Follow conventional commit format
3. Create pull request
4. Automation handles the rest!

## 📚 Additional Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Project Wiki](https://github.com/3C-SCSU/Avatar/wiki)

## 💬 Need Help?

- **Discord**: https://discord.gg/mxbQ7HpPjq
- **Email**: 3c.scsu@gmail.com
- **Issues**: https://github.com/3C-SCSU/Avatar/issues

Happy contributing! 🧠🤖✨
