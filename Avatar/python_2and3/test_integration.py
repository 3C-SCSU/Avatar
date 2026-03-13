#!/usr/bin/env python3
"""
Integration Test for Python 2/3 Environment Setup
Tests that all components work together properly
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class IntegrationTester:
    """Tests integration between Python 2/3 environments"""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.python2_env = self.base_dir / "python2_env"
        self.python3_env = self.base_dir / "python3_env"
        self.avatar_env = self.base_dir / "avatar_env"
        
    def test_environment_isolation(self):
        """Test that environments are properly isolated"""
        print("🧪 Testing Environment Isolation...")
        
        results = {}
        
        # Test Python 3 environment
        python3_bin = self.python3_env / "bin" / "python"
        if python3_bin.exists():
            try:
                result = subprocess.run([
                    str(python3_bin), "-c", 
                    "import sys; print('Python3:', sys.version_info[:2]); import numpy; print('NumPy:', numpy.__version__)"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    results["python3_isolation"] = {
                        "passed": True,
                        "output": result.stdout.strip()
                    }
                    print("✅ Python 3 environment isolation: PASSED")
                else:
                    results["python3_isolation"] = {
                        "passed": False,
                        "error": result.stderr.strip()
                    }
                    print("❌ Python 3 environment isolation: FAILED")
                    
            except subprocess.TimeoutExpired:
                results["python3_isolation"] = {
                    "passed": False,
                    "error": "Timeout"
                }
                print("❌ Python 3 environment isolation: TIMEOUT")
        
        # Test Python 2 environment
        python2_bin = self.python2_env / "bin" / "python"
        if python2_bin.exists():
            try:
                result = subprocess.run([
                    str(python2_bin), "-c",
                    "import sys; print('Python2:', sys.version_info[:2]); import numpy; print('NumPy:', numpy.__version__)"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    results["python2_isolation"] = {
                        "passed": True,
                        "output": result.stdout.strip()
                    }
                    print("✅ Python 2 environment isolation: PASSED")
                else:
                    results["python2_isolation"] = {
                        "passed": False,
                        "error": result.stderr.strip()
                    }
                    print("❌ Python 2 environment isolation: FAILED")
                    
            except subprocess.TimeoutExpired:
                results["python2_isolation"] = {
                    "passed": False,
                    "error": "Timeout"
                }
                print("❌ Python 2 environment isolation: TIMEOUT")
        
        return results
    
    def test_launcher_scripts(self):
        """Test launcher scripts functionality"""
        print("\n🚀 Testing Launcher Scripts...")
        
        results = {}
        
        launcher_tests = [
            ("run_avatar.sh", "Avatar launcher"),
            ("run_nao6.sh", "NAO6 launcher"),
            ("run_drone.sh", "Drone launcher")
        ]
        
        for script_name, description in launcher_tests:
            script_path = self.base_dir / script_name
            
            if script_path.exists():
                # Test script syntax (without actually running)
                try:
                    result = subprocess.run([
                        "bash", "-n", str(script_path)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        results[script_name] = {
                            "passed": True,
                            "description": description
                        }
                        print(f"✅ {description}: Syntax OK")
                    else:
                        results[script_name] = {
                            "passed": False,
                            "error": result.stderr.strip(),
                            "description": description
                        }
                        print(f"❌ {description}: Syntax Error")
                        
                except Exception as e:
                    results[script_name] = {
                        "passed": False,
                        "error": str(e),
                        "description": description
                    }
                    print(f"❌ {description}: Test Failed")
            else:
                results[script_name] = {
                    "passed": False,
                    "error": "Script not found",
                    "description": description
                }
                print(f"❌ {description}: Not Found")
        
        return results
    
    def test_package_compatibility(self):
        """Test package compatibility across environments"""
        print("\n📦 Testing Package Compatibility...")
        
        results = {}
        
        # Test Python 3 packages
        python3_packages = {
            "PySide6": "from PySide6.QtWidgets import QApplication",
            "numpy": "import numpy; print(numpy.__version__)",
            "pandas": "import pandas; print(pandas.__version__)",
            "torch": "import torch; print(torch.__version__)",
            "opencv-python": "import cv2; print(cv2.__version__)"
        }
        
        python3_bin = self.python3_env / "bin" / "python"
        if python3_bin.exists():
            python3_results = {}
            for package, test_code in python3_packages.items():
                try:
                    result = subprocess.run([
                        str(python3_bin), "-c", test_code
                    ], capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0:
                        python3_results[package] = {
                            "passed": True,
                            "output": result.stdout.strip()
                        }
                        print(f"✅ Python 3 {package}: OK")
                    else:
                        python3_results[package] = {
                            "passed": False,
                            "error": result.stderr.strip()
                        }
                        print(f"❌ Python 3 {package}: FAILED")
                        
                except subprocess.TimeoutExpired:
                    python3_results[package] = {
                        "passed": False,
                        "error": "Timeout"
                    }
                    print(f"❌ Python 3 {package}: TIMEOUT")
            
            results["python3_packages"] = python3_results
        
        # Test Python 2 packages
        python2_packages = {
            "qi": "import qi; print('qi imported successfully')",
            "numpy": "import numpy; print(numpy.__version__)",
            "opencv-python": "import cv2; print(cv2.__version__)"
        }
        
        python2_bin = self.python2_env / "bin" / "python"
        if python2_bin.exists():
            python2_results = {}
            for package, test_code in python2_packages.items():
                try:
                    result = subprocess.run([
                        str(python2_bin), "-c", test_code
                    ], capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0:
                        python2_results[package] = {
                            "passed": True,
                            "output": result.stdout.strip()
                        }
                        print(f"✅ Python 2 {package}: OK")
                    else:
                        python2_results[package] = {
                            "passed": False,
                            "error": result.stderr.strip()
                        }
                        print(f"❌ Python 2 {package}: FAILED")
                        
                except subprocess.TimeoutExpired:
                    python2_results[package] = {
                        "passed": False,
                        "error": "Timeout"
                    }
                    print(f"❌ Python 2 {package}: TIMEOUT")
            
            results["python2_packages"] = python2_results
        
        return results
    
    def test_environment_switching(self):
        """Test environment switching functionality"""
        print("\n🔄 Testing Environment Switching...")
        
        results = {}
        
        # Test switch_env.py
        switch_script = self.base_dir / "switch_env.py"
        if switch_script.exists():
            try:
                # Test status command
                result = subprocess.run([
                    sys.executable, str(switch_script), "status"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    results["switch_env_status"] = {
                        "passed": True,
                        "output": result.stdout.strip()
                    }
                    print("✅ Environment switcher status: OK")
                else:
                    results["switch_env_status"] = {
                        "passed": False,
                        "error": result.stderr.strip()
                    }
                    print("❌ Environment switcher status: FAILED")
                    
            except subprocess.TimeoutExpired:
                results["switch_env_status"] = {
                    "passed": False,
                    "error": "Timeout"
                }
                print("❌ Environment switcher status: TIMEOUT")
        else:
            results["switch_env_status"] = {
                "passed": False,
                "error": "Switch script not found"
            }
            print("❌ Environment switcher: NOT FOUND")
        
        return results
    
    def test_cross_environment_communication(self):
        """Test communication between environments"""
        print("\n🔗 Testing Cross-Environment Communication...")
        
        results = {}
        
        # Create a simple test file that can be shared between environments
        test_file = self.base_dir / "test_communication.txt"
        
        try:
            # Test Python 3 writing
            python3_bin = self.python3_env / "bin" / "python"
            if python3_bin.exists():
                result = subprocess.run([
                    str(python3_bin), "-c",
                    f"with open('{test_file}', 'w') as f: f.write('Python3')"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Test Python 2 reading
                    python2_bin = self.python2_env / "bin" / "python"
                    if python2_bin.exists():
                        result = subprocess.run([
                            str(python2_bin), "-c",
                            f"with open('{test_file}', 'r') as f: print(f.read())"
                        ], capture_output=True, text=True)
                        
                        if "Python3" in result.stdout:
                            results["cross_env_comm"] = {
                                "passed": True,
                                "output": "Communication successful"
                            }
                            print("✅ Cross-environment communication: PASSED")
                        else:
                            results["cross_env_comm"] = {
                                "passed": False,
                                "error": "Data mismatch"
                            }
                            print("❌ Cross-environment communication: FAILED")
                    else:
                        results["cross_env_comm"] = {
                            "passed": False,
                            "error": "Python 2 environment not available"
                        }
                        print("❌ Cross-environment communication: PYTHON2 NOT FOUND")
                else:
                    results["cross_env_comm"] = {
                        "passed": False,
                        "error": "Python 3 write failed"
                    }
                    print("❌ Cross-environment communication: PYTHON3 WRITE FAILED")
        
        except Exception as e:
            results["cross_env_comm"] = {
                "passed": False,
                "error": str(e)
            }
            print(f"❌ Cross-environment communication: ERROR - {e}")
        
        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
        
        return results
    
    def run_integration_tests(self):
        """Run all integration tests"""
        print("🧪 Python 2/3 Environment Integration Tests")
        print("=" * 60)
        
        test_results = {
            "environment_isolation": self.test_environment_isolation(),
            "launcher_scripts": self.test_launcher_scripts(),
            "package_compatibility": self.test_package_compatibility(),
            "environment_switching": self.test_environment_switching(),
            "cross_environment_communication": self.test_cross_environment_communication()
        }
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 Integration Test Summary")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in test_results.items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    total_tests += 1
                    if isinstance(result, dict) and result.get("passed", False):
                        passed_tests += 1
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All integration tests passed! Environment is fully functional.")
        else:
            print("\n⚠️ Some integration tests failed. Please check the issues above.")
        
        return test_results
    
    def generate_test_report(self, test_results):
        """Generate a detailed test report"""
        report_file = self.base_dir / "integration_test_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Python 2/3 Environment Integration Test Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category, results in test_results.items():
                f.write(f"## {category.replace('_', ' ').title()}\n\n")
                
                if isinstance(results, dict):
                    for test_name, result in results.items():
                        status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
                        f.write(f"### {test_name}: {status}\n\n")
                        
                        if result.get("output"):
                            f.write(f"**Output:**\n```\n{result['output']}\n```\n\n")
                        
                        if result.get("error"):
                            f.write(f"**Error:**\n```\n{result['error']}\n```\n\n")
        
        print(f"\n📄 Test report generated: {report_file}")

def main():
    """Main integration test function"""
    tester = IntegrationTester()
    
    # Run integration tests
    results = tester.run_integration_tests()
    
    # Generate test report
    tester.generate_test_report(results)
    
    return results

if __name__ == "__main__":
    main()
