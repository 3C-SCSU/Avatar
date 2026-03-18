#!/usr/bin/env python3
"""
Python 2/3 Environment Setup Script for Avatar Project
Solves conflicts between Avatar (Python 3), NAO6 (Python 2), and Tello Drone (Python 3)
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class PythonEnvironmentManager:
    """Manages Python 2 and Python 3 environments for Avatar project"""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.python2_env = self.base_dir / "python2_env"
        self.python3_env = self.base_dir / "python3_env"
        self.avatar_env = self.base_dir / "avatar_env"
        
    def check_python_installations(self):
        """Check available Python installations"""
        print("🔍 Checking Python installations...")
        
        python_versions = {}
        
        # Check Python 3
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            python_versions["python3"] = sys.executable
            print(f"✅ Python 3 found: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Python 3 not found: {e}")
            
        # Check Python 2
        python2_commands = ["python2", "python", "python2.7"]
        for cmd in python2_commands:
            try:
                result = subprocess.run([cmd, "--version"], 
                                      capture_output=True, text=True)
                if "2.7" in result.stdout or "2." in result.stdout:
                    python_versions["python2"] = cmd
                    print(f"✅ Python 2 found: {result.stdout.strip()}")
                    break
            except Exception:
                continue
                
        if "python2" not in python_versions:
            print("❌ Python 2 not found - will install using pyenv")
            
        return python_versions
    
    def install_pyenv_if_needed(self):
        """Install pyenv for Python version management"""
        print("\n📦 Setting up pyenv for Python version management...")
        
        try:
            subprocess.run(["pyenv", "--version"], capture_output=True, check=True)
            print("✅ pyenv already installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📥 Installing pyenv...")
            
            system = platform.system().lower()
            if system == "linux":
                # Install pyenv on Linux
                install_cmd = """
                curl https://pyenv.run | bash
                echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
                echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
                echo 'eval "$(pyenv init -)"' >> ~/.bashrc
                """
                subprocess.run(install_cmd, shell=True, check=True)
            elif system == "darwin":
                # Install pyenv on macOS
                install_cmd = """
                brew install pyenv
                echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
                echo 'eval "$(pyenv init -)"' >> ~/.zshrc
                """
                subprocess.run(install_cmd, shell=True, check=True)
            
            print("✅ pyenv installed successfully")
    
    def install_python2_with_pyenv(self):
        """Install Python 2.7 using pyenv"""
        print("\n📥 Installing Python 2.7 using pyenv...")
        
        try:
            subprocess.run(["pyenv", "install", "2.7.18"], check=True)
            subprocess.run(["pyenv", "global", "2.7.18"], check=True)
            print("✅ Python 2.7.18 installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Python 2.7: {e}")
            return False
    
    def create_virtual_environments(self):
        """Create separate virtual environments for each component"""
        print("\n🏗️ Creating virtual environments...")
        
        # Avatar environment (Python 3)
        print("📦 Creating Avatar environment (Python 3)...")
        self.python3_env.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(self.python3_env)], check=True)
        
        # NAO6 environment (Python 2)
        print("📦 Creating NAO6 environment (Python 2)...")
        self.python2_env.mkdir(exist_ok=True)
        
        # Try to use virtualenv for Python 2
        try:
            subprocess.run(["virtualenv", "-p", "python2.7", str(self.python2_env)], check=True)
            print("✅ NAO6 environment created with virtualenv")
        except subprocess.CalledProcessError:
            print("⚠️ virtualenv not available, creating manual Python 2 environment")
            self.create_manual_python2_env()
        
        # Avatar main environment (Python 3)
        print("📦 Creating Avatar main environment (Python 3)...")
        self.avatar_env.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(self.avatar_env)], check=True)
    
    def create_manual_python2_env(self):
        """Create manual Python 2 environment structure"""
        # Create basic Python 2 environment structure
        (self.python2_env / "bin").mkdir(exist_ok=True)
        (self.python2_env / "lib").mkdir(exist_ok=True)
        
        # Create activation script for Python 2
        activate_script = self.python2_env / "bin" / "activate"
        with open(activate_script, 'w') as f:
            f.write(f"""#!/bin/bash
# Python 2 Environment Activation
export PATH="{self.python2_env}/bin:$PATH"
export PYTHONPATH="{self.python2_env}/lib/python2.7/site-packages:$PYTHONPATH"
export PS1="(python2) $PS1"
echo "✅ Python 2 environment activated"
""")
        activate_script.chmod(0o755)
    
    def install_requirements(self):
        """Install requirements for each environment"""
        print("\n📚 Installing requirements...")
        
        # Avatar Python 3 requirements
        avatar_requirements = self.base_dir / "avatar_requirements.txt"
        if avatar_requirements.exists():
            print("📦 Installing Avatar requirements...")
            pip_path = self.python3_env / "bin" / "pip"
            subprocess.run([str(pip_path), "install", "-r", str(avatar_requirements)], 
                          check=True)
        
        # NAO6 Python 2 requirements
        nao_requirements = self.base_dir / "nao_requirements.txt"
        if nao_requirements.exists():
            print("📦 Installing NAO6 requirements...")
            pip2_path = self.python2_env / "bin" / "pip"
            if pip2_path.exists():
                subprocess.run([str(pip2_path), "install", "-r", str(nao_requirements)], 
                              check=True)
    
    def create_launcher_scripts(self):
        """Create launcher scripts for each component"""
        print("\n🚀 Creating launcher scripts...")
        
        # Avatar launcher
        avatar_launcher = self.base_dir / "run_avatar.sh"
        with open(avatar_launcher, 'w') as f:
            f.write(f"""#!/bin/bash
# Avatar Launcher - Python 3 Environment
echo "🚀 Starting Avatar in Python 3 environment..."
source {self.python3_env}/bin/activate
cd "{self.base_dir.parent.parent}"
python GUI5.py
deactivate
""")
        avatar_launcher.chmod(0o755)
        
        # NAO6 launcher
        nao_launcher = self.base_dir / "run_nao6.sh"
        with open(nao_launcher, 'w') as f:
            f.write(f"""#!/bin/bash
# NAO6 Launcher - Python 2 Environment
echo "🤖 Starting NAO6 in Python 2 environment..."
source {self.python2_env}/bin/activate
cd "{self.base_dir.parent.parent}/NAO6"
python2 nao_controller.py
deactivate
""")
        nao_launcher.chmod(0o755)
        
        # Tello Drone launcher
        drone_launcher = self.base_dir / "run_drone.sh"
        with open(drone_launcher, 'w') as f:
            f.write(f"""#!/bin/bash
# Tello Drone Launcher - Python 3 Environment
echo "🚁 Starting Tello Drone in Python 3 environment..."
source {self.python3_env}/bin/activate
cd "{self.base_dir.parent.parent}"
python GUI5.py
deactivate
""")
        drone_launcher.chmod(0o755)
        
        # Combined launcher
        combined_launcher = self.base_dir / "run_all.sh"
        with open(combined_launcher, 'w') as f:
            f.write(f"""#!/bin/bash
# Combined Avatar System Launcher
echo "🎮 Avatar System Launcher"
echo "========================"
echo "1. Avatar (Python 3)"
echo "2. NAO6 Robot (Python 2)"
echo "3. Tello Drone (Python 3)"
echo "4. All Components"
echo "5. Exit"
echo ""
read -p "Select option [1-5]: " choice

case $choice in
    1)
        echo "🚀 Starting Avatar..."
        {avatar_launcher}
        ;;
    2)
        echo "🤖 Starting NAO6..."
        {nao_launcher}
        ;;
    3)
        echo "🚁 Starting Tello Drone..."
        {drone_launcher}
        ;;
    4)
        echo "🔄 Starting all components..."
        gnome-terminal --tab --title="Avatar" -- {avatar_launcher}
        gnome-terminal --tab --title="NAO6" -- {nao_launcher}
        gnome-terminal --tab --title="Drone" -- {drone_launcher}
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac
""")
        combined_launcher.chmod(0o755)
    
    def create_environment_switcher(self):
        """Create environment switcher utility"""
        print("\n🔄 Creating environment switcher...")
        
        switcher = self.base_dir / "switch_env.py"
        with open(switcher, 'w') as f:
            f.write(f"""#!/usr/bin/env python3
\"\"\"
Environment Switcher Utility
Switch between Python 2 and Python 3 environments
\"\"\"

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
PYTHON2_ENV = BASE_DIR / "python2_env"
PYTHON3_ENV = BASE_DIR / "python3_env"
AVATAR_ENV = BASE_DIR / "avatar_env"

def switch_to_python2():
    \"\"\"Switch to Python 2 environment\"\"\"
    print("🐍 Switching to Python 2 environment...")
    os.environ['PATH'] = f"{PYTHON2_ENV}/bin:" + os.environ.get('PATH', '')
    os.environ['PYTHONPATH'] = f"{PYTHON2_ENV}/lib/python2.7/site-packages:" + os.environ.get('PYTHONPATH', '')
    print("✅ Python 2 environment activated")

def switch_to_python3():
    \"\"\"Switch to Python 3 environment\"\"\"
    print("🐍 Switching to Python 3 environment...")
    os.environ['PATH'] = f"{PYTHON3_ENV}/bin:" + os.environ.get('PATH', '')
    os.environ['PYTHONPATH'] = f"{PYTHON3_ENV}/lib/python3.x/site-packages:" + os.environ.get('PYTHONPATH', '')
    print("✅ Python 3 environment activated")

def show_status():
    \"\"\"Show current environment status\"\"\"
    print("📊 Environment Status:")
    print(f"Python 2 Env: {'✅' if PYTHON2_ENV.exists() else '❌'}")
    print(f"Python 3 Env: {'✅' if PYTHON3_ENV.exists() else '❌'}")
    print(f"Avatar Env: {'✅' if AVATAR_ENV.exists() else '❌'}")
    print(f"Current Python: {sys.version}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "python2":
            switch_to_python2()
        elif sys.argv[1] == "python3":
            switch_to_python3()
        elif sys.argv[1] == "status":
            show_status()
        else:
            print("Usage: python switch_env.py [python2|python3|status]")
    else:
        show_status()
""")
        switcher.chmod(0o755)
    
    def setup_complete(self):
        """Complete setup and provide usage instructions"""
        print("\n" + "="*60)
        print("🎉 Python 2/3 Environment Setup Complete!")
        print("="*60)
        
        print("\n📁 Environments Created:")
        print(f"   • Python 2: {self.python2_env}")
        print(f"   • Python 3: {self.python3_env}")
        print(f"   • Avatar: {self.avatar_env}")
        
        print("\n🚀 Launcher Scripts:")
        print("   • ./run_avatar.sh - Start Avatar (Python 3)")
        print("   • ./run_nao6.sh - Start NAO6 (Python 2)")
        print("   • ./run_drone.sh - Start Tello Drone (Python 3)")
        print("   • ./run_all.sh - Start all components")
        
        print("\n🔧 Utilities:")
        print("   • python switch_env.py status - Show environment status")
        print("   • python switch_env.py python2 - Switch to Python 2")
        print("   • python switch_env.py python3 - Switch to Python 3")
        
        print("\n📋 Next Steps:")
        print("   1. Install component-specific requirements")
        print("   2. Test each environment separately")
        print("   3. Use launcher scripts to run components")
        print("   4. Use switch_env.py for environment management")

def main():
    """Main setup function"""
    print("🔧 Python 2/3 Environment Setup for Avatar Project")
    print("="*60)
    
    manager = PythonEnvironmentManager()
    
    # Check Python installations
    python_versions = manager.check_python_installations()
    
    # Install pyenv if Python 2 not found
    if "python2" not in python_versions:
        manager.install_pyenv_if_needed()
        if not manager.install_python2_with_pyenv():
            print("❌ Cannot proceed without Python 2")
            return False
    
    # Create virtual environments
    manager.create_virtual_environments()
    
    # Install requirements
    manager.install_requirements()
    
    # Create launcher scripts
    manager.create_launcher_scripts()
    
    # Create environment switcher
    manager.create_environment_switcher()
    
    # Setup complete
    manager.setup_complete()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
