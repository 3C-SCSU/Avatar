"""
Avanznow / Tellekynezix - Compliance Automation Tool
----------------------------------------------------
Author: Shariquddin Mohammed
Role: Infrastructure & Compliance Integration
Date: March 2026

Description: Automated batch processor for injecting Apache 2.0 
license boilerplates across the Tellekynezix distributed codebase.
"""


import os

# Standard Apache 2.0 boilerplate for Avanznow / Tellekynezix
APACHE_LICENSE_TEXT = """Copyright 2026 Avanznow / Tellekynezix

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Define comment styles based on file extensions in our BCI/Blockchain stack
COMMENT_STYLES = {
    'hash': {
        'extensions': ['.py', '.sh', '.yml', '.yaml'],
        'format': lambda text: '\n'.join([f"# {line}" if line else "#" for line in text.split('\n')]) + '\n\n'
    },
    'block': {
        'extensions': ['.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts', '.cs', '.go', '.rs'],
        'format': lambda text: "/*\n" + '\n'.join([f" * {line}" if line else " *" for line in text.split('\n')]) + "\n */\n\n"
    }
}

def add_license_to_file(filepath):
    _, ext = os.path.splitext(filepath)
    
    style_to_use = None
    for style_name, style_data in COMMENT_STYLES.items():
        if ext in style_data['extensions']:
            style_to_use = style_data['format']
            break
            
    if not style_to_use:
        return # Skip unsupported or non-source file types

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # OpSec: Check to avoid double-licensing if script is run multiple times
    if "Licensed under the Apache License" in content:
        print(f"Skipped (already licensed): {filepath}")
        return

    header = style_to_use(APACHE_LICENSE_TEXT)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + content)
        print(f"Licensed: {filepath}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        # Exclude hidden directories and dependencies
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv', '.idea']]
        
        for file in files:
            add_license_to_file(os.path.join(root, file))

if __name__ == "__main__":
    target_dir = "." 
    print(f"Initiating Tellekynezix Apache 2.0 License Injection in: {os.path.abspath(target_dir)}")
    process_directory(target_dir)
    print("Injection complete. Ready for commit.")
