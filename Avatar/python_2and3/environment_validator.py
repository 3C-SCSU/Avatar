#!/usr/bin/env python3
"""
Environment Validator for Python 2/3 Setup
Validates that all environments are properly configured
"""

import os
import sys
import subprocess
from pathlib import Path

class EnvironmentValidator:
    """Validates Python 2/3 environments"""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.python2_env = self.base_dir / "python2_env"
        self.python3_env = self.base_dir / "python3_env"
        self.avatar_env = self.base_dir / "avatar_env"
        
    def validate_python_installations(self):
        """Validate Python installations"""
        print("🔍 Validating Python installations...")
        
        results = {}
        
        # Check Python 3
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            results["python3"] = {
                "available": True,
                "version": result.stdout.strip(),
                "path": sys.executable
            }
            print(f"✅ Python 3: {result.stdout.strip()}")
        except Exception as e:
            results["python3"] = {"available": False, "error": str(e)}
            print(f"❌ Python 3: {e}")
        
        # Check Python 2
        python2_commands = ["python2", "python", "python2.7"]
        for cmd in python2_commands:
            try:
                result = subprocess.run([cmd, "--version"], 
                                      capture_output=True, text=True)
                if "2.7" in result.stdout or "2." in result.stdout:
                    results["python2"] = {
                        "available": True,
                        "version": result.stdout.strip(),
                        "path": cmd
                    }
                    print(f"✅ Python 2: {result.stdout.strip()}")
                    break
            except Exception:
                continue
                
        if "python2" not in results:
            print("❌ Python 2 not found")
            results["python2"] = {"available": False, "error": "Not found"}
            
        return results
    
    def validate_virtual_environments(self):
        """Validate virtual environments"""
        print("\n🏗️ Validating virtual environments...")
        
        environments = {
            "python2_env": self.python2_env,
            "python3_env": self.python3_env,
            "avatar_env": self.avatar_env
        }
        
        results = {}
        
        for name, path in environments.items():
            if path.exists():
                # Check for bin directory and activation script
                bin_dir = path / "bin"
                activate_script = bin_dir / "activate"
                
                if bin_dir.exists() and activate_script.exists():
                    results[name] = {
                        "exists": True,
                        "valid": True,
                        "path": str(path)
                    }
                    print(f"✅ {name}: Valid environment")
                else:
                    results[name] = {
                        "exists": True,
                        "valid": False,
                        "path": str(path),
                        "error": "Missing activation script"
                    }
                    print(f"⚠️ {name}: Incomplete environment")
            else:
                results[name] = {
                    "exists": False,
                    "valid": False,
                    "path": str(path),
                    "error": "Environment not found"
                }
                print(f"❌ {name}: Not found")
        
        return results
    
    def validate_package_installations(self):
        """Validate package installations in each environment"""
        print("\n📦 Validating package installations...")
        
        results = {}
        
        # Python 3 packages
        python3_packages = [
            "PySide6", "numpy", "pandas", "torch", 
            "opencv-python", "matplotlib", "djitellopy"
        ]
        
        python2_packages = [
            "qi", "numpy", "opencv-python", "Pillow", "scikit-learn"
        ]
        
        # Test Python 3 environment
        if self.python3_env.exists():
            python3_bin = self.python3_env / "bin"
            if python3_bin.exists():
                python3_exe = python3_bin / "python"
                if python3_exe.exists():
                    results["python3_packages"] = self._check_packages(
                        str(python3_exe), python3_packages
                    )
        
        # Test Python 2 environment
        if self.python2_env.exists():
            python2_bin = self.python2_env / "bin"
            if python2_bin.exists():
                python2_exe = python2_bin / "python"
                if python2_exe.exists():
                    results["python2_packages"] = self._check_packages(
                        str(python2_exe), python2_packages
                    )
        
        return results
    
    def _check_packages(self, python_exe, packages):
        """Check if packages are installed in given Python environment"""
        results = {}
        
        for package in packages:
            try:
                result = subprocess.run([
                    python_exe, "-c", f"import {package}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    results[package] = {"installed": True}
                    print(f"✅ {package}: Installed")
                else:
                    results[package] = {"installed": False}
                    print(f"❌ {package}: Not installed")
                    
            except Exception as e:
                results[package] = {"installed": False, "error": str(e)}
                print(f"❌ {package}: Error - {e}")
        
        return results
    
    def validate_launcher_scripts(self):
        """Validate launcher scripts"""
        print("\n🚀 Validating launcher scripts...")
        
        launcher_scripts = [
            "run_avatar.sh", "run_nao6.sh", "run_drone.sh", "run_all.sh"
        ]
        
        results = {}
        
        for script in launcher_scripts:
            script_path = self.base_dir / script
            if script_path.exists():
                # Check if script is executable
                if os.access(script_path, os.X_OK):
                    results[script] = {
                        "exists": True,
                        "executable": True,
                        "path": str(script_path)
                    }
                    print(f"✅ {script}: Executable")
                else:
                    results[script] = {
                        "exists": True,
                        "executable": False,
                        "path": str(script_path),
                        "error": "Not executable"
                    }
                    print(f"⚠️ {script}: Not executable")
            else:
                results[script] = {
                    "exists": False,
                    "executable": False,
                    "path": str(script_path),
                    "error": "Not found"
                }
                print(f"❌ {script}: Not found")
        
        return results
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation of all environments"""
        print("🔧 Python 2/3 Environment Validation")
        print("=" * 50)
        
        validation_results = {
            "python_installations": self.validate_python_installations(),
            "virtual_environments": self.validate_virtual_environments(),
            "package_installations": self.validate_package_installations(),
            "launcher_scripts": self.validate_launcher_scripts()
        }
        
        # Generate summary
        print("\n" + "=" * 50)
        print("📊 Validation Summary")
        print("=" * 50)
        
        total_checks = 0
        passed_checks = 0
        
        for category, results in validation_results.items():
            if isinstance(results, dict):
                for item, result in results.items():
                    total_checks += 1
                    if isinstance(result, dict):
                        if result.get("valid", result.get("available", result.get("installed", False))):
                            passed_checks += 1
        
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
        
        if passed_checks == total_checks:
            print("\n🎉 All validations passed! Environment is ready.")
        else:
            print("\n⚠️ Some validations failed. Please check the issues above.")
        
        return validation_results
    
    def generate_fix_script(self, validation_results):
        """Generate a script to fix common issues"""
        fix_script = self.base_dir / "fix_environment.sh"
        
        with open(fix_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated environment fix script\n\n")
            
            # Fix Python 2 installation
            python_installations = validation_results.get("python_installations", {})
            if not python_installations.get("python2", {}).get("available", False):
                f.write("# Install Python 2.7\n")
                f.write("echo 'Installing Python 2.7...'\n")
                f.write("curl https://pyenv.run | bash\n")
                f.write("pyenv install 2.7.18\n\n")
            
            # Fix virtual environments
            virtual_envs = validation_results.get("virtual_environments", {})
            for env_name, env_result in virtual_envs.items():
                if not env_result.get("valid", False):
                    f.write(f"# Recreate {env_name}\n")
                    f.write(f"echo 'Recreating {env_name}...'\n")
                    if "python2" in env_name:
                        f.write(f"virtualenv -p python2.7 {env_result['path']}\n")
                    else:
                        f.write(f"python3 -m venv {env_result['path']}\n")
                    f.write("\n")
            
            # Fix launcher scripts
            launchers = validation_results.get("launcher_scripts", {})
            for script_name, script_result in launchers.items():
                if not script_result.get("executable", False):
                    f.write(f"# Make {script_name} executable\n")
                    f.write(f"chmod +x {script_result['path']}\n\n")
            
            f.write("echo 'Environment fixes completed!'\n")
        
        fix_script.chmod(0o755)
        print(f"🔧 Fix script generated: {fix_script}")

def main():
    """Main validation function"""
    validator = EnvironmentValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Generate fix script if needed
    failed_checks = sum(
        1 for category in results.values()
        for item in category.values()
        if isinstance(item, dict) and not item.get("valid", item.get("available", item.get("installed", False)))
    )
    
    if failed_checks > 0:
        print(f"\n🔧 Generating fix script for {failed_checks} issues...")
        validator.generate_fix_script(results)
    
    return results

if __name__ == "__main__":
    main()
