#!/bin/bash
# Setup script for Avatar BCI project automation
# This script configures conventional commits and tests automation locally

set -e

echo "🤖 Avatar BCI Project - Automation Setup"
echo "========================================"

# Check if we're in the right directory
if [[ ! -f "VERSION" || ! -f "CHANGELOG.md" ]]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📁 Detected project root: $(pwd)"

# Configure git commit template
echo "📝 Setting up git commit template..."
if git config commit.template .gitcommit-template; then
    echo "✅ Git commit template configured"
    echo "   Now when you run 'git commit', you'll see helpful prompts"
else
    echo "⚠️  Warning: Could not configure git commit template"
fi

# Install Python dependencies for automation
echo "🐍 Installing Python dependencies..."
if python3 -m pip install GitPython semantic-version > /dev/null 2>&1; then
    echo "✅ Python dependencies installed"
else
    echo "⚠️  Warning: Could not install Python dependencies"
    echo "   You may need to run: pip install GitPython semantic-version"
fi

# Test automation script
echo "🧪 Testing automation script..."
if python3 scripts/auto_version.py > /dev/null 2>&1; then
    echo "✅ Automation script working correctly"
else
    echo "⚠️  Warning: Automation script test had issues"
    echo "   Try running manually: python3 scripts/auto_version.py"
fi

# Show current version
echo "📊 Current project status:"
echo "   Version: $(cat VERSION)"
echo "   Last commit: $(git log -1 --oneline)"

# Check GitHub Actions workflow
if [[ -f ".github/workflows/release.yml" ]]; then
    echo "✅ GitHub Actions workflow configured"
else
    echo "⚠️  Warning: GitHub Actions workflow not found"
fi

echo ""
echo "🎉 Setup complete! Here's what you can do now:"
echo ""
echo "📚 Learn conventional commits:"
echo "   📖 Read CONTRIBUTING.md for detailed examples"
echo "   🔗 Visit: https://www.conventionalcommits.org/"
echo ""
echo "🚀 Start contributing:"
echo "   git commit -m 'feat(bci): add amazing new feature'"
echo "   git commit -m 'fix(drone): resolve connection issue'"
echo "   git commit -m 'docs: update README with new info'"
echo ""
echo "🧪 Test automation locally:"
echo "   python3 scripts/auto_version.py"
echo ""
echo "📋 Useful commands:"
echo "   git log --oneline -10          # See recent commits"
echo "   cat VERSION                     # Check current version"
echo "   git status                      # Check working directory"
echo ""
echo "Happy contributing! 🧠🤖✨"
