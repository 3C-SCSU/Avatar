#!/usr/bin/env python3
"""
Automatic Version and Changelog Management

This script automatically:
1. Analyzes git commits using conventional commit format
2. Determines appropriate version bump (major, minor, patch)
3. Updates VERSION file
4. Updates CHANGELOG.md with new entries
5. Follows semantic versioning and Keep a Changelog format
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import git
    import semantic_version
except ImportError:
    print("❌ Required dependencies not found.")
    print("📦 Please install them with:")
    print("   pip install GitPython semantic-version")
    print("   OR run: ./scripts/setup_automation.sh")
    sys.exit(1)


class ConventionalCommit:
    """Parse and analyze conventional commits"""
    
    # Commit type mappings to version bumps
    TYPE_MAPPING = {
        'feat': 'minor',        # New features
        'feature': 'minor',     # New features (alternative)
        'fix': 'patch',         # Bug fixes
        'bugfix': 'patch',      # Bug fixes (alternative)
        'hotfix': 'patch',      # Critical bug fixes
        'perf': 'patch',        # Performance improvements
        'performance': 'patch', # Performance improvements (alternative)
        'docs': 'patch',        # Documentation changes
        'documentation': 'patch', # Documentation changes (alternative)
        'style': 'patch',       # Code style changes
        'refactor': 'patch',    # Code refactoring
        'refactoring': 'patch', # Code refactoring (alternative)
        'test': 'patch',        # Test additions/changes
        'tests': 'patch',       # Test additions/changes (alternative)
        'chore': 'patch',       # Maintenance tasks
        'ci': 'patch',          # CI/CD changes
        'cd': 'patch',          # CI/CD changes (alternative)
        'build': 'patch',       # Build system changes
        'deps': 'patch',        # Dependency updates
        'dependencies': 'patch', # Dependency updates (alternative)
        'security': 'patch',    # Security fixes
        'sec': 'patch',         # Security fixes (alternative)
        'deprecated': 'minor',  # Deprecation notices
        'deprecate': 'minor',   # Deprecation notices (alternative)
        'removed': 'major',     # Removed features
        'remove': 'major',      # Removed features (alternative)
    }
    
    # Changelog section mappings
    CHANGELOG_MAPPING = {
        'feat': 'Added',
        'feature': 'Added',
        'fix': 'Fixed',
        'bugfix': 'Fixed', 
        'hotfix': 'Fixed',
        'perf': 'Changed',
        'performance': 'Changed',
        'docs': 'Changed',
        'documentation': 'Changed',
        'style': 'Changed',
        'refactor': 'Changed',
        'refactoring': 'Changed',
        'test': 'Changed',
        'tests': 'Changed',
        'chore': 'Changed',
        'ci': 'Changed',
        'cd': 'Changed',
        'build': 'Changed',
        'deps': 'Changed',
        'dependencies': 'Changed',
        'security': 'Security',
        'sec': 'Security',
        'deprecated': 'Deprecated',
        'deprecate': 'Deprecated',
        'removed': 'Removed',
        'remove': 'Removed',
    }

    def __init__(self, commit_message: str, commit_hash: str = ""):
        self.raw_message = commit_message
        self.commit_hash = commit_hash
        self.type = None
        self.scope = None
        self.description = ""
        self.body = ""
        self.breaking_change = False
        self.parse()

    def parse(self):
        """Parse conventional commit format"""
        lines = self.raw_message.strip().split('\n')
        header = lines[0] if lines else ""
        
        # Parse header: type(scope): description
        pattern = r'^(\w+)(?:\(([^)]+)\))?(!)?:\s*(.+)$'
        match = re.match(pattern, header)
        
        if match:
            self.type = match.group(1)
            self.scope = match.group(2)
            self.breaking_change = bool(match.group(3))
            self.description = match.group(4)
        else:
            # Fallback: treat as chore if not conventional format
            self.type = 'chore'
            self.description = header

        # Check for breaking changes in body
        self.body = '\n'.join(lines[1:]).strip()
         # Match "BREAKING CHANGE:" / "BREAKING-CHANGE:" (case-insensitive), line-anchored
        if re.search(r'(?im)^(?:BREAKING(?:\s|-)?CHANGES?):', self.body):
            self.breaking_change = True

    def get_version_bump(self) -> str:
        """Determine version bump type"""
        if self.breaking_change:
            return 'major'
        return self.TYPE_MAPPING.get(self.type, 'patch')

    def get_changelog_section(self) -> str:
        """Get appropriate changelog section"""
        return self.CHANGELOG_MAPPING.get(self.type, 'Changed')

    def format_for_changelog(self) -> str:
        """Format commit for changelog entry"""
        scope_text = f"({self.scope})" if self.scope else ""
        breaking_text = " [BREAKING CHANGE]" if self.breaking_change else ""
        hash_text = f" ({self.commit_hash[:7]})" if self.commit_hash else ""
        
        return f"- {self.description.capitalize()}{scope_text}{breaking_text}{hash_text}"


class VersionManager:
    """Manage version updates and changelog generation"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.version_file = self.repo_path / "VERSION"
        self.changelog_file = self.repo_path / "CHANGELOG.md"
        self.repo = git.Repo(repo_path)

    def get_current_version(self) -> semantic_version.Version:
        """Read current version from VERSION file"""
        if not self.version_file.exists():
            return semantic_version.Version("0.0.0")
        
        version_text = self.version_file.read_text().strip()
        return semantic_version.Version(version_text)

    def get_commits_since_last_release(self) -> List[git.Commit]:
        """Get commits since last version tag"""
        try:
            # Try to find the last version tag
            tags = sorted(self.repo.tags, key=lambda t: t.commit.committed_date, reverse=True)
            last_tag = None
            
            for tag in tags:
                if re.match(r'^v?\d+\.\d+\.\d+', tag.name):
                    last_tag = tag
                    break
            
            if last_tag:
                commit_range = f"{last_tag.commit.hexsha}..HEAD"
                commits = list(self.repo.iter_commits(commit_range))
            else:
                # No version tags found, get all commits
                commits = list(self.repo.iter_commits("HEAD", max_count=50))
            
            # Filter out merge commits and release commits
            filtered_commits = []
            for commit in commits:
                if (not commit.message.startswith("Merge") and 
                    "[skip ci]" not in commit.message and
                    not commit.message.startswith("chore: release")):
                    filtered_commits.append(commit)
            
            return filtered_commits
            
        except Exception as e:
            print(f"Warning: Could not get commits since last release: {e}")
            return []

    def determine_version_bump(self, commits: List[git.Commit]) -> str:
        """Analyze commits to determine version bump"""
        if not commits:
            return "patch"  # Default to patch if no commits
        
        bump_types = []
        for commit in commits:
            conv_commit = ConventionalCommit(commit.message, commit.hexsha)
            bump_types.append(conv_commit.get_version_bump())
        
        # Priority: major > minor > patch
        if 'major' in bump_types:
            return 'major'
        elif 'minor' in bump_types:
            return 'minor'
        else:
            return 'patch'

    def bump_version(self, current_version: semantic_version.Version, bump_type: str) -> semantic_version.Version:
        """Bump version according to semantic versioning"""
        if bump_type == 'major':
            return current_version.next_major()
        elif bump_type == 'minor':
            return current_version.next_minor()
        else:
            return current_version.next_patch()

    def generate_changelog_entries(self, commits: List[git.Commit]) -> Dict[str, List[str]]:
        """Generate changelog entries grouped by type"""
        entries = {
            'Added': [],
            'Changed': [],
            'Deprecated': [],
            'Removed': [],
            'Fixed': [],
            'Security': []
        }
        
        for commit in commits:
            conv_commit = ConventionalCommit(commit.message, commit.hexsha)
            section = conv_commit.get_changelog_section()
            entry = conv_commit.format_for_changelog()
            
            if section in entries:
                entries[section].append(entry)
        
        # Remove empty sections
        return {k: v for k, v in entries.items() if v}

    def update_changelog(self, new_version: semantic_version.Version, entries: Dict[str, List[str]]):
        """Update CHANGELOG.md with new entries"""
        if not entries:
            print("No changelog entries to add")
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        new_section = f"\n## [{new_version}] - {today}\n\n"
        
        for section, items in entries.items():
            if items:
                new_section += f"### {section}\n"
                for item in items:
                    new_section += f"{item}\n"
                new_section += "\n"
        
        # Read current changelog
        if self.changelog_file.exists():
            content = self.changelog_file.read_text()
            
            # Find the insertion point (after [Unreleased])
            unreleased_pattern = r'(## \[Unreleased\]\s*\n)'
            if re.search(unreleased_pattern, content):
                content = re.sub(unreleased_pattern, f'\\1{new_section}', content)
            else:
                # Fallback: insert after the header
                lines = content.split('\n')
                insert_index = 8  # After standard header
                lines.insert(insert_index, new_section.strip())
                content = '\n'.join(lines)
        else:
            # Create new changelog
            content = self.create_initial_changelog(new_version, entries)
        
        self.changelog_file.write_text(content)
        print(f"Updated {self.changelog_file}")

    def create_release_notes(self, version: semantic_version.Version, entries: Dict[str, List[str]]):
        """Create release notes file for GitHub release"""
        if not entries:
            return
        
        # Map changelog sections to release note categories
        release_categories = {
            'Fixed': 'FIXES',
            'Added': 'NEW FEATURES', 
            'Changed': 'CHANGES',
            'Deprecated': 'DEPRECATED',
            'Removed': 'REMOVED',
            'Security': 'SECURITY'
        }
        
        release_notes = f"# Release v{version}\n\n"

        # Derive repo/branch for links
        repo_slug = os.environ.get("GITHUB_REPOSITORY", "")
        ref_name = os.environ.get("GITHUB_REF_NAME", "main")
        base_url = ""
        if repo_slug:
            base_url = f"https://github.com/{repo_slug}/blob/{ref_name}"
        else:
            # Fallback: parse origin remote
            try:
                origin_url = next((r.url for r in self.repo.remotes if r.name == "origin"), "")
                m = re.search(r"github\.com[:/](.*?)(?:\.git)?$", origin_url)
                if m:
                    base_url = f"https://github.com/{m.group(1)}/blob/{ref_name}"
            except Exception:
                pass
        
        # Order categories by importance
        category_order = ['FIXES', 'NEW FEATURES', 'CHANGES', 'SECURITY', 'DEPRECATED', 'REMOVED']
        
        for category in category_order:
            # Find matching entries for this category
            category_items = []
            for section, items in entries.items():
                if release_categories.get(section) == category and items:
                    category_items.extend(items)
            
            if category_items:
                release_notes += f"## {category}:\n"
                for item in category_items:
                    # Remove the leading "- " and format consistently
                    clean_item = item[2:] if item.startswith("- ") else item
                    release_notes += f" - {clean_item}\n"
                release_notes += "\n"
        
        release_notes += "---\n\n"

        if base_url:
            release_notes += f"**Full Changelog**: [CHANGELOG.md]({base_url}/CHANGELOG.md)\n"
            release_notes += f"**Installation**: See [SETUP_GUIDE.md]({base_url}/SETUP_GUIDE.md)\n"
        else:
            release_notes += "**Full Changelog**: CHANGELOG.md\n"
            release_notes += "**Installation**: See SETUP_GUIDE.md\n"

        # Write to release notes file
        release_notes_file = self.repo_path / "RELEASE_NOTES.md"
        release_notes_file.write_text(release_notes)
        print(f"Created release notes: {release_notes_file}")

    def create_initial_changelog(self, version: semantic_version.Version, entries: Dict[str, List[str]]) -> str:
        """Create initial changelog file"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        content = """# Changelog

All notable changes to the Avatar BCI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

"""
        
        content += f"## [{version}] - {today}\n\n"
        
        for section, items in entries.items():
            if items:
                content += f"### {section}\n"
                for item in items:
                    content += f"{item}\n"
                content += "\n"
        
        return content

    def update_version_file(self, new_version: semantic_version.Version):
        """Update VERSION file"""
        self.version_file.write_text(str(new_version))
        print(f"Updated version to {new_version}")

    def run_automation(self):
        """Main automation workflow"""
        print("🤖 Starting automated version and changelog update...")
        
        # Get current state
        current_version = self.get_current_version()
        commits = self.get_commits_since_last_release()
        
        print(f"📋 Current version: {current_version}")
        print(f"📝 Found {len(commits)} commits since last release")
        
        if not commits:
            print("✅ No new commits found. Nothing to update.")
            return
        
        # Analyze and determine updates
        bump_type = self.determine_version_bump(commits)
        new_version = self.bump_version(current_version, bump_type)
        changelog_entries = self.generate_changelog_entries(commits)
        
        print(f"📈 Version bump: {bump_type} ({current_version} → {new_version})")
        
        # Apply updates
        self.update_version_file(new_version)
        self.update_changelog(new_version, changelog_entries)
        self.create_release_notes(new_version, changelog_entries)
        
        print("✅ Automation completed successfully!")
        print(f"🎉 New version: {new_version}")


def main():
    """Main entry point"""
    try:
        manager = VersionManager()
        manager.run_automation()
    except Exception as e:
        print(f"❌ Error during automation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
