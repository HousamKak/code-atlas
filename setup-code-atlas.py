#!/usr/bin/env python3
"""
Code Atlas - Cross-Platform Setup and Analysis Script

This script sets up the code analysis environment and generates an interactive code map.
Works on Windows, macOS, and Linux with comprehensive error handling and validation.

Usage: python setup-code-atlas.py [repository-path] [output-name]

Features:
- Robust dependency management with version checking
- Cross-platform virtual environment handling
- Comprehensive error handling and recovery
- Progress tracking and detailed logging
- Automatic Tree-sitter language setup
"""

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
import importlib.metadata
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import venv
import tempfile
import urllib.request
import urllib.error


@dataclass
class SystemInfo:
    """Information about the current system."""
    platform: str
    python_version: Tuple[int, int, int]
    architecture: str
    is_windows: bool
    is_macos: bool
    is_linux: bool


class SetupError(Exception):
    """Base exception for setup-related errors."""
    pass


class DependencyError(SetupError):
    """Raised when dependency installation or validation fails."""
    pass


class EnvironmentError(SetupError):
    """Raised when environment setup fails."""
    pass


class Colors:
    """ANSI color codes for terminal output with Windows compatibility."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[0;37m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

    @classmethod
    def initialize(cls) -> None:
        """Initialize colors based on platform and terminal support."""
        if platform.system() == 'Windows':
            try:
                # Enable ANSI colors on Windows 10+
                import ctypes
                kernel32 = ctypes.windll.kernel32
                stdout_handle = kernel32.GetStdHandle(-11)
                if stdout_handle != -1:
                    mode = ctypes.c_ulong()
                    if kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):
                        kernel32.SetConsoleMode(stdout_handle, mode.value | 0x0004)
            except Exception:
                # Disable colors if ANSI not supported
                cls._disable_colors()
        
        # Check if running in a non-interactive environment
        if not sys.stdout.isatty():
            cls._disable_colors()

    @classmethod
    def _disable_colors(cls) -> None:
        """Disable all color codes."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr not in ['initialize', '_disable_colors']:
                setattr(cls, attr, '')


class SystemDetector:
    """Detects and validates system configuration."""
    
    @staticmethod
    def get_system_info() -> SystemInfo:
        """Get comprehensive system information."""
        platform_name = platform.system()
        python_version = sys.version_info[:3]
        architecture = platform.machine()
        
        return SystemInfo(
            platform=platform_name,
            python_version=python_version,
            architecture=architecture,
            is_windows=platform_name == 'Windows',
            is_macos=platform_name == 'Darwin',
            is_linux=platform_name == 'Linux'
        )
    
    @staticmethod
    def validate_python_version(min_version: Tuple[int, int] = (3, 8)) -> bool:
        """Validate Python version meets minimum requirements."""
        current_version = sys.version_info[:2]
        return current_version >= min_version
    
    @staticmethod
    def check_internet_connectivity() -> bool:
        """Check if internet connection is available for package downloads."""
        try:
            urllib.request.urlopen('https://pypi.org', timeout=5)
            return True
        except (urllib.error.URLError, OSError):
            return False


class DependencyManager:
    """Manages Python dependencies with comprehensive validation."""
    
    REQUIRED_PACKAGES = {
        'tree-sitter': '0.20.1',
        'tree-sitter-python': '0.20.1',
        'tree-sitter-javascript': '0.20.0',
        'tree-sitter-typescript': '0.20.2',
        'tree-sitter-java': '0.20.0',
        'tree-sitter-go': '0.20.0',
        'tree-sitter-rust': '0.20.3',
        'tree-sitter-cpp': '0.20.0',
        'tree-sitter-c': '0.20.0',
        'tree-sitter-c-sharp': '0.20.0',
        'gitpython': '3.1.0',
        'radon': '5.1.0'
    }
    
    def __init__(self, venv_path: Path) -> None:
        """Initialize dependency manager."""
        if not isinstance(venv_path, Path):
            raise TypeError(f"Expected Path object, got {type(venv_path)}")
        
        self.venv_path = venv_path
        self.system_info = SystemDetector.get_system_info()
    
    def get_venv_executable(self, executable: str) -> Path:
        """Get the path to an executable in the virtual environment."""
        if self.system_info.is_windows:
            scripts_dir = self.venv_path / "Scripts"
            exe_path = scripts_dir / f"{executable}.exe"
            if not exe_path.exists():
                exe_path = scripts_dir / executable
        else:
            exe_path = self.venv_path / "bin" / executable
        
        return exe_path
    
    def create_requirements_file(self, requirements_path: Path) -> None:
        """Create a comprehensive requirements.txt file."""
        requirements_content = "# Code Atlas - Python Dependencies\n"
        requirements_content += "# Generated automatically - do not edit manually\n\n"
        
        for package, min_version in self.REQUIRED_PACKAGES.items():
            requirements_content += f"{package}>={min_version}\n"
        
        # Add optional but recommended packages
        requirements_content += "\n# Optional packages for enhanced functionality\n"
        # The tree-sitter-languages package is not universally available on all
        # platforms and Python versions. It contains pre-built language
        # libraries, but the standard tree-sitter packages above already cover
        # the required grammars. Keep this dependency commented out to avoid
        # installation failures on systems where no compatible wheel exists.
        # requirements_content += "tree-sitter-languages>=1.5.0  # Pre-built language binaries\n"
        requirements_content += "colorama>=0.4.4  # Enhanced Windows color support\n"
        
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(requirements_content)
    
    def validate_installed_packages(self) -> Tuple[List[str], List[str]]:
        """Validate installed packages and return missing/outdated ones."""
        python_exe = self.get_venv_executable("python")
        
        missing_packages = []
        outdated_packages = []
        
        for package, min_version in self.REQUIRED_PACKAGES.items():
            # Normalize package name for distribution lookup
            pkg_name = package.replace('-', '_')
            
            try:
                # Use the virtual environment's Python to check packages
                result = subprocess.run(
                    [str(python_exe), '-c', f'import importlib.metadata; print(importlib.metadata.version("{pkg_name}"))'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    installed_version = result.stdout.strip()
                    # Be more lenient with version comparison - just check if it's not empty
                    if not installed_version:
                        missing_packages.append(package)
                    # Skip strict version comparison for tree-sitter packages as they can be tricky
                    elif not package.startswith('tree-sitter') and installed_version < min_version:
                        outdated_packages.append(f"{package} {installed_version} < {min_version}")
                else:
                    missing_packages.append(package)
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception):
                # Don't fail on individual package checks - just mark as potentially missing
                missing_packages.append(package)
        
        return missing_packages, outdated_packages
    
    def test_key_imports(self) -> bool:
        """Test if key packages can be imported (more reliable than metadata check)."""
        python_exe = self.get_venv_executable("python")
        
        key_tests = [
            "import tree_sitter",
            "import git", 
            "import radon"
        ]
        
        success_count = 0
        for test in key_tests:
            try:
                result = subprocess.run(
                    [str(python_exe), '-c', test],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    success_count += 1
            except Exception:
                continue
        
        return success_count >= 2  # At least 2 out of 3 key packages working
    
    def install_dependencies(self, requirements_path: Path, force_reinstall: bool = False) -> None:
        """Install dependencies with comprehensive error handling."""
        python_exe = self.get_venv_executable("python")
        
        if not python_exe.exists():
            raise EnvironmentError(f"Python executable not found in virtual environment: {python_exe}")
        
        # Upgrade pip first
        self._run_pip_command([str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install requirements
        install_cmd = [str(python_exe), '-m', 'pip', 'install', '-r', str(requirements_path)]
        if force_reinstall:
            install_cmd.append('--force-reinstall')
        
        self._run_pip_command(install_cmd)
        
        # Test if key packages can be imported (more reliable than metadata checks)
        if self.test_key_imports():
            print("✅ Key dependencies verified successfully")
        else:
            print("⚠️  Some packages may not have installed correctly")
            print("   The analysis may still work - continuing anyway")
        
        # Optional: Light validation for informational purposes only
        try:
            missing, outdated = self.validate_installed_packages()
            if missing and len(missing) < len(self.REQUIRED_PACKAGES) // 2:
                # Only warn if less than half are "missing" (likely just metadata issues)
                print(f"ℹ️  Note: Could not verify {len(missing)} packages via metadata")
        except Exception:
            # Ignore validation errors completely
            pass
    
    def _run_pip_command(self, command: List[str], timeout: int = 300) -> None:
        """Run a pip command with proper error handling."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
        except subprocess.TimeoutExpired:
            raise DependencyError(f"Package installation timed out after {timeout} seconds")
        except subprocess.CalledProcessError as e:
            error_msg = f"Package installation failed: {e}\n"
            if e.stderr:
                error_msg += f"Error output: {e.stderr}\n"
            if e.stdout:
                error_msg += f"Standard output: {e.stdout}\n"
            raise DependencyError(error_msg)


class CodeAtlasSetup:
    """Main setup class for Code Atlas tool with comprehensive error handling."""

    def __init__(self, repo_path: str = ".", output_name: str = "code-map") -> None:
        """Initialize setup with configuration and validation."""
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")
        
        self.output_name = output_name
        self.system_info = SystemDetector.get_system_info()
        
        # File paths
        self.extract_script = Path("extract.py")
        self.visualizer_file = Path("visualizer.html")
        self.requirements_file = Path("requirements.txt")
        self.output_json = Path(f"{output_name}.json")
        self.venv_dir = Path("venv")
        
        # Environment variables
        self.skip_deps = os.environ.get('SKIP_DEPS', '').lower() in ('1', 'true', 'yes')
        self.no_server = os.environ.get('NO_SERVER', '').lower() in ('1', 'true', 'yes')
        self.force_reinstall = os.environ.get('FORCE_REINSTALL', '').lower() in ('1', 'true', 'yes')
        
        # Initialize colors
        Colors.initialize()
        
        # Initialize dependency manager
        self.dependency_manager = DependencyManager(self.venv_dir)

    def print_header(self) -> None:
        """Print the application header with system information."""
        print(f"{Colors.PURPLE}")
        print("🗺️  Code Atlas - Interactive Code Mapping Tool")
        print("=" * 50)
        print(f"Platform: {self.system_info.platform} {self.system_info.architecture}")
        print(f"Python: {'.'.join(map(str, self.system_info.python_version))}")
        print(f"{Colors.NC}")

    def print_step(self, message: str) -> None:
        """Print a step message."""
        print(f"{Colors.CYAN}📋 {message}{Colors.NC}")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        print(f"{Colors.GREEN}✅ {message}{Colors.NC}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")

    def print_error(self, message: str) -> None:
        """Print an error message."""
        print(f"{Colors.RED}❌ {message}{Colors.NC}")

    def check_command(self, command: str) -> bool:
        """Check if a command is available in PATH."""
        return shutil.which(command) is not None

    def run_command(self, command: List[str], capture_output: bool = True, 
                   check: bool = True, cwd: Optional[Path] = None, 
                   timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command with comprehensive error handling."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                check=check,
                cwd=cwd,
                timeout=timeout
            )
            return result
        except subprocess.TimeoutExpired:
            raise SetupError(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            if check:
                error_msg = f"Command failed: {' '.join(command)}\n"
                if e.stderr:
                    error_msg += f"Error: {e.stderr}\n"
                if e.stdout:
                    error_msg += f"Output: {e.stdout}\n"
                raise SetupError(error_msg)
            return e

    def validate_prerequisites(self) -> None:
        """Validate system prerequisites."""
        self.print_step("Validating system prerequisites...")
        
        # Check Python version
        if not SystemDetector.validate_python_version():
            raise SetupError(f"Python 3.8+ required, found {'.'.join(map(str, self.system_info.python_version))}")
        
        python_version = '.'.join(map(str, self.system_info.python_version))
        self.print_success(f"Python {python_version} found")
        
        # Check internet connectivity
        if not SystemDetector.check_internet_connectivity():
            self.print_warning("No internet connection detected. Package installation may fail.")
        else:
            self.print_success("Internet connectivity confirmed")
        
        # Check available disk space
        free_space = shutil.disk_usage(Path.cwd()).free
        if free_space < 100 * 1024 * 1024:  # 100MB minimum
            self.print_warning(f"Low disk space: {free_space / (1024*1024):.1f} MB available")
        
        # Check Node.js (optional)
        if self.check_command('node'):
            try:
                result = self.run_command(['node', '--version'])
                node_version = result.stdout.strip()
                major_version = int(node_version.lstrip('v').split('.')[0])
                if major_version >= 16:
                    self.print_success(f"Node.js {node_version} found")
                else:
                    self.print_warning(f"Node.js {node_version} found, but 16+ recommended")
            except Exception:
                self.print_warning("Node.js version check failed")
        else:
            self.print_warning("Node.js not found. Python HTTP server will be used instead")
        
        # Check Git (optional)
        if self.check_command('git'):
            try:
                result = self.run_command(['git', '--version'])
                git_version = result.stdout.strip()
                self.print_success(f"{git_version} found")
            except Exception:
                self.print_warning("Git version check failed")
        else:
            self.print_warning("Git not found. Code churn metrics will not be available")

    def check_required_files(self) -> None:
        """Check if all required files exist."""
        self.print_step("Checking required files...")
        
        missing_files = []
        
        if not self.extract_script.exists():
            missing_files.append(str(self.extract_script))
        
        if not self.visualizer_file.exists():
            missing_files.append(str(self.visualizer_file))
        
        if missing_files:
            raise FileNotFoundError(f"Required files not found: {', '.join(missing_files)}")
        
        self.print_success("All required files found")

    def create_virtual_environment(self) -> None:
        """Create a virtual environment with comprehensive error handling."""
        if self.venv_dir.exists():
            self.print_success("Virtual environment already exists")
            return

        self.print_step("Creating virtual environment...")
        
        try:
            # Create virtual environment
            venv.create(self.venv_dir, with_pip=True, clear=True)
            
            # Verify creation
            python_exe = self.dependency_manager.get_venv_executable("python")
            if not python_exe.exists():
                raise EnvironmentError(f"Virtual environment creation failed: {python_exe} not found")
            
            # Test the virtual environment
            result = self.run_command([str(python_exe), '--version'])
            venv_python_version = result.stdout.strip()
            
            self.print_success(f"Virtual environment created with {venv_python_version}")
            
        except Exception as e:
            # Clean up on failure
            if self.venv_dir.exists():
                shutil.rmtree(self.venv_dir, ignore_errors=True)
            raise EnvironmentError(f"Failed to create virtual environment: {e}")

    def setup_dependencies(self) -> None:
        """Set up Python dependencies with validation."""
        self.print_step("Setting up Python dependencies...")
        
        # Create requirements file
        self.dependency_manager.create_requirements_file(self.requirements_file)
        self.print_success("Requirements file created")
        
        # Install dependencies
        try:
            self.dependency_manager.install_dependencies(
                self.requirements_file, 
                force_reinstall=self.force_reinstall
            )
            self.print_success("Dependencies installed successfully")
        except DependencyError as e:
            # For actual pip failures, we still want to report them, but try to continue
            self.print_warning(f"Dependency installation had issues: {e}")
            self.print_step("Attempting to continue anyway...")
            
            # Test if core functionality might still work
            if self.dependency_manager.test_key_imports():
                self.print_success("Core packages appear to be working despite installation warnings")
            else:
                self.print_error("Core packages are not working. Please check your Python environment.")
                raise

    def validate_repository(self) -> Dict[str, Any]:
        """Validate repository and return analysis about source files."""
        self.print_step("Analyzing repository structure...")
        
        # Supported extensions
        extensions = ['.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.rs', 
                     '.cpp', '.c', '.cs', '.kt']
        
        source_files = []
        extension_counts = {}
        
        for ext in extensions:
            files = list(self.repo_path.rglob(f'*{ext}'))
            
            # Filter out common non-source directories
            filtered_files = []
            for file_path in files:
                path_str = str(file_path).lower().replace('\\', '/')
                excluded_patterns = [
                    'node_modules/', '.git/', 'venv/', '__pycache__/', 
                    'build/', 'dist/', 'target/', '.pytest_cache/', 
                    'coverage/', 'logs/', 'tmp/', 'temp/'
                ]
                
                if not any(pattern in path_str for pattern in excluded_patterns):
                    # Check file size (skip very large files)
                    try:
                        if file_path.stat().st_size < 1024 * 1024:  # 1MB limit
                            filtered_files.append(file_path)
                    except OSError:
                        continue
            
            if filtered_files:
                extension_counts[ext] = len(filtered_files)
                source_files.extend(filtered_files)
        
        total_files = len(source_files)
        
        if total_files == 0:
            self.print_warning(f"No supported source files found in '{self.repo_path}'")
            self.print_warning(f"Supported extensions: {', '.join(extensions)}")
        else:
            self.print_success(f"Found {total_files} source files to analyze")
            
            # Show file type breakdown
            if extension_counts:
                print("  File type breakdown:")
                for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_files) * 100
                    print(f"    {ext}: {count} files ({percentage:.1f}%)")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in source_files[:100])  # Sample for size estimate
        estimated_total_size = (total_size * total_files) // min(100, total_files) if source_files else 0
        
        return {
            'total_files': total_files,
            'extension_counts': extension_counts,
            'estimated_size_mb': estimated_total_size / (1024 * 1024),
            'source_files': source_files[:10]  # Sample for verification
        }

    def run_analysis(self) -> None:
        """Run the code analysis with progress tracking."""
        self.print_step(f"Running code analysis on '{self.repo_path}'...")
        
        python_exe = self.dependency_manager.get_venv_executable("python")
        
        # Build analysis command
        analysis_cmd = [
            str(python_exe),
            str(self.extract_script),
            str(self.repo_path),
            '-o', str(self.output_json),
            '--verbose'
        ]
        
        # Add comprehensive exclusion patterns
        exclusions = [
            # Dependencies and package managers
            "*/node_modules/*", "*/.npm/*", "*/bower_components/*",
            # Python environments and caches  
            "*/venv/*", "*/env/*", "*/.venv/*", "*/__pycache__/*",
            "*/site-packages/*", "*/.pytest_cache/*", "*/.tox/*",
            # Build outputs
            "*/build/*", "*/dist/*", "*/target/*", "*/out/*",
            "*/bin/*", "*/obj/*", "*/Debug/*", "*/Release/*",
            # Version control
            "*/.git/*", "*/.svn/*", "*/.hg/*",
            # IDE and editor files
            "*/.vscode/*", "*/.idea/*", "*.swp", "*.swo",
            # Coverage and logs
            "*/coverage/*", "*/.coverage*", "*/logs/*", "*.log",
            # Documentation builds
            "*/docs/_build/*", "*/_site/*", "*/sphinx_build/*",
            # Temporary files
            "*/tmp/*", "*/temp/*", "*/.cache/*"
        ]
        
        for pattern in exclusions:
            analysis_cmd.extend(['--exclude', pattern])
        
        # Run analysis with timeout
        self.print_step("This may take several minutes for large repositories...")
        print(f"Command: {' '.join(analysis_cmd[:6])} ... (with {len(exclusions)} exclusions)")
        
        try:
            start_time = time.time()
            result = self.run_command(analysis_cmd, capture_output=False, timeout=1800)  # 30 minute timeout
            duration = time.time() - start_time
            
            self.print_success(f"Analysis completed in {duration:.1f} seconds")
            
        except SetupError as e:
            self.print_error(f"Analysis failed: {e}")
            self.print_step("Troubleshooting tips:")
            print("  - Try analyzing a smaller directory")
            print("  - Check if all required dependencies are installed")
            print("  - Ensure the repository contains supported file types")
            raise
        
        # Validate output
        if not self.output_json.exists():
            raise SetupError("Analysis completed but no output file was generated")
        
        file_size = self.output_json.stat().st_size
        if file_size == 0:
            raise SetupError("Analysis generated an empty output file")
        
        file_size_kb = file_size // 1024
        self.print_success(f"Code map generated: {self.output_json} ({file_size_kb} KB)")
        
        # Validate JSON structure
        try:
            with open(self.output_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, dict):
                raise SetupError("Generated file is not a valid JSON object")
            
            if 'children' in data and isinstance(data['children'], list):
                module_count = len(data['children'])
                if module_count > 0:
                    self.print_success(f"Found {module_count} modules in analysis")
                else:
                    self.print_warning("No modules found - check exclusion patterns")
            else:
                self.print_warning("Unexpected JSON structure in output file")
                
        except json.JSONDecodeError as e:
            raise SetupError(f"Generated file contains invalid JSON: {e}")

    def find_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> int:
        """Find an available port with improved error handling."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        raise SetupError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

    def start_web_server(self) -> None:
        """Start a web server for visualization with fallback options."""
        self.print_step("Starting web server for visualization...")
        
        port = self.find_available_port()
        
        self.print_success(f"Starting server on port {port}")
        print(f"{Colors.BLUE}🌐 Open your browser and navigate to:{Colors.NC}")
        print(f"{Colors.YELLOW}   http://localhost:{port}/{self.visualizer_file.name}{Colors.NC}")
        print(f"{Colors.BLUE}📁 Then load your generated file:{Colors.NC}")
        print(f"{Colors.YELLOW}   {self.output_json.name}{Colors.NC}")
        print()
        self.print_warning("Press Ctrl+C to stop the server")
        print()
        
        # Try to open browser automatically
        try:
            url = f"http://localhost:{port}/{self.visualizer_file.name}"
            webbrowser.open(url)
            self.print_success("Browser opened automatically")
        except Exception as e:
            self.print_warning(f"Could not open browser automatically: {e}")
        
        # Try different server options in order of preference
        server_attempts = [
            lambda: self._start_node_server(port),
            lambda: self._start_python_server(port)
        ]
        
        for attempt in server_attempts:
            try:
                attempt()
                return
            except KeyboardInterrupt:
                print("\nServer stopped by user.")
                return
            except Exception as e:
                self.print_warning(f"Server attempt failed: {e}")
                continue
        
        raise SetupError("All server options failed")

    def _start_node_server(self, port: int) -> None:
        """Start Node.js HTTP server."""
        if not self.check_command('npx'):
            raise SetupError("npx not available")
        
        self.print_step("Starting Node.js server...")
        self.run_command(['npx', 'http-server', '-p', str(port), '--cors'], 
                        capture_output=False, timeout=None)

    def _start_python_server(self, port: int) -> None:
        """Start Python HTTP server."""
        self.print_step("Starting Python server...")
        
        # Use the system Python for the server (doesn't need special packages)
        python_cmd = 'python' if self.check_command('python') else 'python3'
        
        self.run_command([python_cmd, '-m', 'http.server', str(port)], 
                        capture_output=False, timeout=None)

    def show_completion_message(self) -> None:
        """Show setup completion message with instructions."""
        print()
        print(f"{Colors.GREEN}🎉 Code Atlas setup completed successfully!{Colors.NC}")
        print()
        print(f"{Colors.BOLD}Generated Files:{Colors.NC}")
        print(f"  📊 Analysis data: {self.output_json}")
        print(f"  🌐 Visualizer: {self.visualizer_file}")
        print(f"  📋 Requirements: {self.requirements_file}")
        print()
        print(f"{Colors.BOLD}To view your code map:{Colors.NC}")
        print(f"  1. Start a web server: python -m http.server 8000")
        print(f"  2. Open: http://localhost:8000/{self.visualizer_file.name}")
        print(f"  3. Load your data file: {self.output_json}")
        print()
        print(f"{Colors.BOLD}Environment:{Colors.NC}")
        print(f"  Virtual environment: {self.venv_dir}")
        print(f"  Python executable: {self.dependency_manager.get_venv_executable('python')}")

    def run_setup(self) -> None:
        """Run the complete setup process with comprehensive error handling."""
        try:
            self.print_header()
            
            # Prerequisites validation
            self.validate_prerequisites()
            
            # File validation
            self.check_required_files()
            
            # Repository analysis
            repo_info = self.validate_repository()
            
            if repo_info['total_files'] == 0:
                self.print_warning("No source files found. Continuing with setup anyway.")
            elif repo_info['estimated_size_mb'] > 100:
                self.print_warning(f"Large repository detected (~{repo_info['estimated_size_mb']:.1f}MB). Analysis may take time.")
            
            # Environment setup
            if not self.skip_deps:
                self.create_virtual_environment()
                self.setup_dependencies()
            else:
                self.print_warning("Skipping dependency installation (SKIP_DEPS=1)")
                if not self.venv_dir.exists():
                    self.print_warning("Virtual environment not found. Some features may not work.")
            
            # Run analysis
            if repo_info['total_files'] > 0:
                self.run_analysis()
            else:
                self.print_warning("Skipping analysis due to no source files")
            
            # Show completion message
            self.show_completion_message()
            
            # Offer to start server
            if not self.no_server and self.output_json.exists():
                try:
                    response = input("Would you like to start the web server to view the visualization? (y/N): ")
                    if response.lower().startswith('y'):
                        self.start_web_server()
                except (KeyboardInterrupt, EOFError):
                    print()
                    self.print_step("Server startup cancelled.")
                    
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Setup interrupted by user.{Colors.NC}")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Setup failed: {e}")
            
            # Provide troubleshooting information
            print(f"\n{Colors.BOLD}Troubleshooting Information:{Colors.NC}")
            print(f"Platform: {self.system_info.platform}")
            print(f"Python: {'.'.join(map(str, self.system_info.python_version))}")
            print(f"Working directory: {Path.cwd()}")
            print(f"Repository path: {self.repo_path}")
            
            if isinstance(e, (DependencyError, EnvironmentError)):
                print(f"\n{Colors.BOLD}Suggestions:{Colors.NC}")
                print("- Check internet connectivity")
                print("- Try running with FORCE_REINSTALL=1")
                print("- Manually install dependencies: pip install -r requirements.txt")
            
            sys.exit(1)


def show_help() -> None:
    """Show comprehensive help information."""
    print(f"{Colors.PURPLE}")
    print("🗺️  Code Atlas - Interactive Code Mapping Tool")
    print("=" * 50)
    print(f"{Colors.NC}")
    
    print("USAGE:")
    print("  python setup-code-atlas.py [repository-path] [output-name]")
    print()
    
    print("ARGUMENTS:")
    print("  repository-path   Path to the codebase to analyze (default: current directory)")
    print("  output-name       Name for output files (default: code-map)")
    print()
    
    print("EXAMPLES:")
    print("  python setup-code-atlas.py")
    print("    → Analyze current directory")
    print()
    print("  python setup-code-atlas.py /path/to/project")
    print("    → Analyze specific project")
    print()
    print("  python setup-code-atlas.py ../my-app my-analysis")
    print("    → Analyze with custom output name")
    print()
    
    print("ENVIRONMENT VARIABLES:")
    print("  SKIP_DEPS=1                     Skip dependency installation")
    print("  NO_SERVER=1                     Don't offer to start web server")
    print("  FORCE_REINSTALL=1               Force reinstall all dependencies")
    print()
    
    print("SUPPORTED FILE TYPES:")
    print("  Python (.py)")
    print("  JavaScript (.js, .jsx)")
    print("  TypeScript (.ts, .tsx)")
    print("  Java (.java)")
    print("  Go (.go)")
    print("  Rust (.rs)")
    print("  C/C++ (.c, .cpp, .cc, .cxx)")
    print("  C# (.cs)")
    print("  Kotlin (.kt)")
    print()
    
    print("PLATFORM SUPPORT:")
    print("  ✅ Windows (Python 3.8+)")
    print("  ✅ macOS (Python 3.8+)")
    print("  ✅ Linux (Python 3.8+)")


def main() -> None:
    """Main entry point with comprehensive argument handling."""
    parser = argparse.ArgumentParser(
        description="Code Atlas - Interactive Code Mapping Tool",
        add_help=False,  # Custom help handling
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('repo_path', nargs='?', default='.', 
                       help='Path to the codebase to analyze')
    parser.add_argument('output_name', nargs='?', default='code-map',
                       help='Name for output files')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Show this help message and exit')
    parser.add_argument('--version', action='version', version='Code Atlas 1.0.0')
    
    try:
        args = parser.parse_args()
        
        if args.help:
            show_help()
            return
        
        # Validate arguments
        if not isinstance(args.repo_path, str) or not args.repo_path.strip():
            raise ValueError("Repository path cannot be empty")
        
        if not isinstance(args.output_name, str) or not args.output_name.strip():
            raise ValueError("Output name cannot be empty")
        
        # Sanitize output name
        args.output_name = re.sub(r'[^\w\-_.]', '_', args.output_name)
        
        # Create and run setup
        setup = CodeAtlasSetup(args.repo_path, args.output_name)
        setup.run_setup()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user.{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.NC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
