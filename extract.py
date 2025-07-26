#!/usr/bin/env python3
"""
Code Map Data Extractor - Fixed Tree-sitter Compatibility

A comprehensive tool for parsing codebases and extracting hierarchical metrics
for visualization as interactive code maps.

This version includes fixes for Tree-sitter API compatibility with newer versions.

Dependencies:
    tree-sitter>=0.20.1
    tree-sitter-python>=0.20.1
    tree-sitter-javascript>=0.20.0
    tree-sitter-typescript>=0.20.2
    tree-sitter-java>=0.20.0
    gitpython>=3.1.0
    radon>=5.1.0
"""

import json
import os
import re
import subprocess
import sys
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Protocol, cast
from collections import defaultdict, Counter
import argparse
import logging
import importlib.metadata
from abc import ABC, abstractmethod

# Type definitions for better code clarity
TreeSitterNode = Any  # Tree-sitter node type
TreeSitterTree = Any  # Tree-sitter tree type
GitCommit = Any  # Git commit type


class DependencyError(Exception):
    """Raised when required dependencies are missing or incompatible."""
    pass


class LanguageSetupError(Exception):
    """Raised when Tree-sitter language setup fails."""
    pass


class AnalysisError(Exception):
    """Raised when code analysis encounters unrecoverable errors."""
    pass


def validate_dependencies() -> None:
    """Validate that all required dependencies are available with correct versions."""
    required_packages = {
        'tree-sitter': '0.20.1',
        'gitpython': '3.1.0'
    }
    
    missing_packages = []
    version_mismatches = []
    
    for package, min_version in required_packages.items():
        try:
            installed_version = importlib.metadata.version(package.replace('-', '_'))
            if installed_version < min_version:
                version_mismatches.append(f"{package} {installed_version} < {min_version}")
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)
    
    if missing_packages or version_mismatches:
        error_msg = "Dependency issues found:\n"
        if missing_packages:
            error_msg += f"Missing packages: {', '.join(missing_packages)}\n"
        if version_mismatches:
            error_msg += f"Version mismatches: {', '.join(version_mismatches)}\n"
        error_msg += "Please install with: pip install -r requirements.txt"
        raise DependencyError(error_msg)


# Import with proper error handling
try:
    validate_dependencies()
    import tree_sitter
    from tree_sitter import Language, Parser
except ImportError as e:
    raise DependencyError(f"tree-sitter not found: {e}. Install with: pip install tree-sitter")

try:
    import git
    from git import Repo
except ImportError:
    git = None
    logging.warning("GitPython not found. Git metrics will be disabled.")

try:
    from radon.complexity import cc_visit
    from radon.raw import analyze as radon_analyze
except ImportError:
    logging.warning("Radon not found. Advanced complexity metrics will be limited.")
    cc_visit = None
    radon_analyze = None


@dataclass
class CallInfo:
    """Information about function/method calls with validation."""
    incoming: int = field(default=0)
    outgoing: int = field(default=0)
    
    def __post_init__(self) -> None:
        """Validate call counts are non-negative."""
        if self.incoming < 0:
            raise ValueError(f"Incoming calls cannot be negative: {self.incoming}")
        if self.outgoing < 0:
            raise ValueError(f"Outgoing calls cannot be negative: {self.outgoing}")


@dataclass
class CodeNode:
    """Represents a node in the code hierarchy (module, class, function)."""
    name: str
    node_type: str  # 'module', 'class', 'function', 'method', 'root'
    loc: int = field(default=0)
    complexity: int = field(default=0)
    coverage: float = field(default=0.0)
    churn: int = field(default=0)
    call_count: Optional[CallInfo] = field(default=None)
    children: Optional[List['CodeNode']] = field(default=None)
    file_path: str = field(default="")
    start_line: int = field(default=0)
    end_line: int = field(default=0)

    def __post_init__(self) -> None:
        """Initialize default values and validate inputs."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Node name must be a non-empty string: {self.name}")
        
        valid_types = {'module', 'class', 'function', 'method', 'root', 'interface'}
        if self.node_type not in valid_types:
            raise ValueError(f"Invalid node type: {self.node_type}. Must be one of {valid_types}")
        
        if self.loc < 0:
            raise ValueError(f"Lines of code cannot be negative: {self.loc}")
        
        if self.complexity < 0:
            raise ValueError(f"Complexity cannot be negative: {self.complexity}")
            
        if not 0.0 <= self.coverage <= 100.0:
            raise ValueError(f"Coverage must be between 0 and 100: {self.coverage}")
        
        if self.churn < 0:
            raise ValueError(f"Churn cannot be negative: {self.churn}")
        
        if self.call_count is None:
            self.call_count = CallInfo()
        if self.children is None:
            self.children = []
            
        if self.start_line < 0:
            raise ValueError(f"Start line cannot be negative: {self.start_line}")
        if self.end_line < 0:
            raise ValueError(f"End line cannot be negative: {self.end_line}")
        if self.start_line > 0 and self.end_line > 0 and self.start_line > self.end_line:
            raise ValueError(f"Start line {self.start_line} cannot be greater than end line {self.end_line}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format suitable for JSON serialization."""
        if self.call_count is None:
            raise ValueError("CallInfo cannot be None when converting to dict")
            
        result = {
            "name": self.name,
            "type": self.node_type,
            "loc": self.loc,
            "complexity": self.complexity,
            "coverage": self.coverage,
            "churn": self.churn,
            "callCount": {
                "in": self.call_count.incoming, 
                "out": self.call_count.outgoing
            }
        }
        
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
            
        return result


class LanguageParser(Protocol):
    """Protocol for language-specific parsers."""
    
    def parse_file(self, source_code: str) -> TreeSitterTree:
        """Parse source code and return syntax tree."""
        ...
    
    def extract_definitions(self, tree: TreeSitterTree, source_code: str) -> List[Tuple[str, str, int, int]]:
        """Extract function/class definitions from syntax tree."""
        ...


class TreeSitterSetup:
    """Manages Tree-sitter language setup and parsing with robust error handling."""
    
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'javascript',
        '.java': 'java',
        '.kt': 'kotlin',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.cs': 'c_sharp'
    }

    def __init__(self) -> None:
        """Initialize Tree-sitter setup with error handling."""
        self.languages: Dict[str, Language] = {}
        self.parsers: Dict[str, Parser] = {}
        self.failed_languages: Set[str] = set()
        self._setup_languages()

    def _setup_languages(self) -> None:
        """Set up Tree-sitter languages and parsers with fallback mechanisms."""
        unique_languages = set(self.SUPPORTED_LANGUAGES.values())
        
        for lang_name in unique_languages:
            try:
                self._setup_single_language(lang_name)
            except Exception as e:
                self.failed_languages.add(lang_name)
                logging.warning(f"Failed to load Tree-sitter language '{lang_name}': {e}")
        
        if not self.languages:
            raise LanguageSetupError(
                "No Tree-sitter languages could be loaded. Please check your installation."
            )
        
        successful_languages = list(self.languages.keys())
        logging.info(f"Successfully loaded Tree-sitter languages: {successful_languages}")
        
        if self.failed_languages:
            logging.warning(f"Failed to load languages: {sorted(self.failed_languages)}")

    def _setup_single_language(self, lang_name: str) -> None:
        """Set up a single Tree-sitter language with multiple fallback attempts."""
        # Attempt different approaches to load the language
        load_attempts = [
            lambda: self._load_from_installed_package(lang_name),
            lambda: self._load_from_system_library(lang_name),
            lambda: self._load_from_build_directory(lang_name),
        ]
        
        last_error = None
        for attempt in load_attempts:
            try:
                language = attempt()
                if language is not None:
                    parser = Parser()
                    parser.set_language(language)
                    
                    self.languages[lang_name] = language
                    self.parsers[lang_name] = parser
                    logging.info(f"Successfully loaded Tree-sitter language: {lang_name}")
                    return
            except Exception as e:
                last_error = e
                continue
        
        raise LanguageSetupError(f"All loading attempts failed for {lang_name}: {last_error}")

    def _load_from_installed_package(self, lang_name: str) -> Optional[Language]:
        """Attempt to load language from installed Python package (preferred method)."""
        try:
            # Map language names to package names
            package_map = {
                'python': 'tree_sitter_python',
                'javascript': 'tree_sitter_javascript', 
                'typescript': 'tree_sitter_typescript',
                'java': 'tree_sitter_java',
                'go': 'tree_sitter_go',
                'rust': 'tree_sitter_rust',
                'cpp': 'tree_sitter_cpp',
                'c': 'tree_sitter_c',
                'c_sharp': 'tree_sitter_c_sharp',
                'kotlin': 'tree_sitter_kotlin'
            }
            
            package_name = package_map.get(lang_name)
            if not package_name:
                return None
                
            # Try to import the language package
            try:
                package = __import__(package_name)
                # Get the language function - this varies by package
                if hasattr(package, 'language'):
                    language_func = package.language
                elif hasattr(package, f'{lang_name}_language'):
                    language_func = getattr(package, f'{lang_name}_language')
                else:
                    # Look for any function that returns a Language
                    for attr_name in dir(package):
                        attr = getattr(package, attr_name)
                        if callable(attr) and not attr_name.startswith('_'):
                            try:
                                result = attr()
                                if isinstance(result, Language):
                                    language_func = attr
                                    break
                            except:
                                continue
                    else:
                        return None
                
                return language_func()
                
            except ImportError:
                return None
                
        except Exception as e:
            logging.debug(f"Failed to load {lang_name} from package: {e}")
            return None

    def _load_from_system_library(self, lang_name: str) -> Optional[Language]:
        """Attempt to load language from system library (legacy method)."""
        try:
            lib_name = f"tree-sitter-{lang_name.replace('_', '-')}"
            # Try the newer API first (single argument)
            return Language(lib_name)
        except TypeError:
            try:
                # Fall back to older API (two arguments)
                return Language(lib_name, lang_name)
            except Exception:
                return None
        except Exception:
            return None

    def _load_from_build_directory(self, lang_name: str) -> Optional[Language]:
        """Attempt to load language from build directory."""
        try:
            build_path = Path("build") / f"{lang_name}.so"
            if build_path.exists():
                # Try newer API first
                try:
                    return Language(str(build_path))
                except TypeError:
                    # Fall back to older API
                    return Language(str(build_path), lang_name)
        except Exception:
            pass
        return None

    def get_parser(self, file_extension: str) -> Optional[Parser]:
        """Get parser for given file extension with validation."""
        if not isinstance(file_extension, str):
            raise TypeError(f"File extension must be string, got {type(file_extension)}")
            
        lang_name = self.SUPPORTED_LANGUAGES.get(file_extension.lower())
        if lang_name and lang_name in self.parsers:
            return self.parsers[lang_name]
        return None

    def get_language_name(self, file_extension: str) -> Optional[str]:
        """Get language name for given file extension with validation."""
        if not isinstance(file_extension, str):
            raise TypeError(f"File extension must be string, got {type(file_extension)}")
            
        return self.SUPPORTED_LANGUAGES.get(file_extension.lower())

    def is_supported(self, file_extension: str) -> bool:
        """Check if file extension is supported and language is loaded."""
        lang_name = self.get_language_name(file_extension)
        return lang_name is not None and lang_name in self.parsers


class MetricsCalculator:
    """Calculates various code metrics with robust error handling."""

    def __init__(self, repo_path: Optional[str] = None) -> None:
        """Initialize metrics calculator with validation."""
        self.repo_path = repo_path
        self.git_repo: Optional[Repo] = None
        
        if repo_path:
            if not isinstance(repo_path, str):
                raise TypeError(f"Repository path must be string, got {type(repo_path)}")
            
            repo_path_obj = Path(repo_path)
            if not repo_path_obj.exists():
                raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
            
            if git:
                try:
                    self.git_repo = Repo(repo_path)
                except git.exc.InvalidGitRepositoryError:
                    logging.warning(f"Not a git repository: {repo_path}")
                except Exception as e:
                    logging.warning(f"Failed to initialize git repository: {e}")

    def calculate_loc(self, source_code: str) -> int:
        """Calculate lines of code (excluding empty lines and comments)."""
        if not isinstance(source_code, str):
            raise TypeError(f"Expected string, got {type(source_code)}")
        
        if not source_code.strip():
            return 0
            
        lines = source_code.strip().split('\n')
        loc = 0
        
        for line in lines:
            stripped = line.strip()
            # More comprehensive comment detection
            if (stripped and 
                not stripped.startswith('#') and 
                not stripped.startswith('//') and
                not stripped.startswith('/*') and
                not stripped.startswith('*') and
                not stripped == '*/'):
                loc += 1
                
        return loc

    def calculate_complexity(self, source_code: str, language: str) -> int:
        """Calculate cyclomatic complexity with validation."""
        if not isinstance(source_code, str):
            raise TypeError(f"Expected string, got {type(source_code)}")
        if not isinstance(language, str):
            raise TypeError(f"Language must be string, got {type(language)}")
        
        if not source_code.strip():
            return 1  # Base complexity for empty code
            
        # For Python, use radon if available
        if language == 'python' and cc_visit:
            try:
                complexity_data = cc_visit(source_code)
                total_complexity = sum(item.complexity for item in complexity_data)
                return max(1, total_complexity)  # Minimum complexity of 1
            except Exception as e:
                logging.debug(f"Radon complexity calculation failed: {e}")
        
        # Fallback: count control flow statements
        complexity_keywords = {
            'python': ['if', 'elif', 'while', 'for', 'except', 'finally', 'with', 'and', 'or'],
            'javascript': ['if', 'else', 'while', 'for', 'switch', 'case', 'catch', 'finally', '&&', '||'],
            'typescript': ['if', 'else', 'while', 'for', 'switch', 'case', 'catch', 'finally', '&&', '||'],
            'java': ['if', 'else', 'while', 'for', 'switch', 'case', 'catch', 'finally', '&&', '||'],
        }
        
        keywords = complexity_keywords.get(language, ['if', 'while', 'for'])
        complexity = 1  # Base complexity
        
        for keyword in keywords:
            if keyword in ['&&', '||']:
                complexity += source_code.count(keyword)
            else:
                complexity += len(re.findall(rf'\b{keyword}\b', source_code))
            
        return complexity

    def calculate_churn(self, file_path: str, days: int = 90) -> int:
        """Calculate git churn (number of commits) for a file with validation."""
        if not isinstance(file_path, str):
            raise TypeError(f"File path must be string, got {type(file_path)}")
        if not isinstance(days, int) or days <= 0:
            raise ValueError(f"Days must be positive integer, got {days}")
            
        if not self.git_repo:
            return 0
            
        try:
            since_date = f"--since={days} days ago"
            commits = list(self.git_repo.iter_commits(since_date, paths=file_path))
            return len(commits)
        except Exception as e:
            logging.debug(f"Git churn calculation failed for {file_path}: {e}")
            return 0


class CodeAnalyzer:
    """Main code analysis engine with comprehensive error handling."""

    def __init__(self, repo_path: str) -> None:
        """Initialize code analyzer with validation."""
        if not isinstance(repo_path, str):
            raise TypeError(f"Expected string path, got {type(repo_path)}")
            
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")
            
        try:
            self.tree_sitter = TreeSitterSetup()
        except LanguageSetupError as e:
            raise AnalysisError(f"Failed to initialize Tree-sitter: {e}")
            
        self.metrics = MetricsCalculator(str(self.repo_path))
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Track function/class definitions and their calls
        self.definitions: Dict[str, CodeNode] = {}
        self.call_references: Dict[str, List[str]] = defaultdict(list)

    def _extract_node_info(self, node: TreeSitterNode, source_code: str, language: str) -> Optional[Tuple[str, str, int, int]]:
        """Extract information from a Tree-sitter node."""
        node_types = {
            'python': {
                'function_definition': 'function',
                'async_function_definition': 'function',
                'class_definition': 'class',
                'module': 'module'
            },
            'javascript': {
                'function_declaration': 'function',
                'function_expression': 'function',
                'arrow_function': 'function',
                'class_declaration': 'class',
                'method_definition': 'method'
            },
            'typescript': {
                'function_declaration': 'function',
                'function_expression': 'function',
                'arrow_function': 'function',
                'class_declaration': 'class',
                'method_definition': 'method',
                'interface_declaration': 'interface'
            },
            'java': {
                'method_declaration': 'method',
                'constructor_declaration': 'method',
                'class_declaration': 'class',
                'interface_declaration': 'interface'
            }
        }
        
        lang_types = node_types.get(language, {})
        node_type = lang_types.get(node.type)
        
        if not node_type:
            return None
            
        # Extract node name
        name = self._get_node_name(node, source_code, language)
        if not name:
            return None
            
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        return name, node_type, start_line, end_line

    def _get_node_name(self, node: TreeSitterNode, source_code: str, language: str) -> Optional[str]:
        """Extract the name of a function, class, or method."""
        # Look for identifier child nodes
        for child in node.children:
            if child.type == 'identifier':
                name = source_code[child.start_byte:child.end_byte]
                if name and name.isidentifier():  # Validate identifier
                    return name
        return None

    def _extract_calls(self, node: TreeSitterNode, source_code: str, language: str) -> List[str]:
        """Extract function/method calls from a node."""
        calls = []
        
        call_types = {
            'python': ['call'],
            'javascript': ['call_expression'],
            'typescript': ['call_expression'],
            'java': ['method_invocation']
        }
        
        target_types = call_types.get(language, ['call', 'call_expression'])
        
        def traverse(n: TreeSitterNode) -> None:
            if n.type in target_types:
                # Extract the function name being called
                for child in n.children:
                    if child.type in ['identifier', 'attribute']:
                        call_name = source_code[child.start_byte:child.end_byte]
                        if call_name and call_name.replace('.', '_').replace('_', '').isalnum():
                            calls.append(call_name)
                        break
            
            for child in n.children:
                traverse(child)
        
        try:
            traverse(node)
        except Exception as e:
            logging.debug(f"Failed to extract calls: {e}")
        
        return calls

    def analyze_file(self, file_path: Path) -> CodeNode:
        """Analyze a single source file with comprehensive error handling."""
        if not isinstance(file_path, Path):
            raise TypeError(f"Expected Path object, got {type(file_path)}")
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
            
        extension = file_path.suffix
        parser = self.tree_sitter.get_parser(extension)
        language = self.tree_sitter.get_language_name(extension)
        
        if not parser or not language:
            logging.warning(f"Unsupported file type: {extension}")
            return CodeNode(
                name=file_path.stem, 
                node_type='module', 
                file_path=str(file_path.relative_to(self.repo_path))
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            return CodeNode(
                name=file_path.stem, 
                node_type='module', 
                file_path=str(file_path.relative_to(self.repo_path))
            )
        
        try:
            tree = parser.parse(bytes(source_code, 'utf8'))
            root_node = tree.root_node
        except Exception as e:
            logging.error(f"Failed to parse file {file_path}: {e}")
            return CodeNode(
                name=file_path.stem, 
                node_type='module', 
                file_path=str(file_path.relative_to(self.repo_path)),
                loc=self.metrics.calculate_loc(source_code)
            )
        
        # Create module node
        relative_path = str(file_path.relative_to(self.repo_path))
        module_node = CodeNode(
            name=file_path.stem,
            node_type='module',
            file_path=relative_path,
            loc=self.metrics.calculate_loc(source_code),
            complexity=self.metrics.calculate_complexity(source_code, language),
            churn=self.metrics.calculate_churn(relative_path)
        )
        
        # Extract classes and functions
        try:
            self._extract_hierarchy(root_node, source_code, language, module_node)
        except Exception as e:
            logging.error(f"Failed to extract hierarchy from {file_path}: {e}")
        
        # Calculate call counts
        try:
            self._calculate_call_counts(module_node)
        except Exception as e:
            logging.error(f"Failed to calculate call counts for {file_path}: {e}")
        
        return module_node

    def _extract_hierarchy(self, node: TreeSitterNode, source_code: str, language: str, parent: CodeNode) -> None:
        """Recursively extract code hierarchy with error handling."""
        if not hasattr(node, 'children'):
            return
            
        for child in node.children:
            try:
                node_info = self._extract_node_info(child, source_code, language)
                
                if node_info:
                    name, node_type, start_line, end_line = node_info
                    
                    # Extract code segment for this node
                    lines = source_code.split('\n')
                    if start_line <= len(lines) and end_line <= len(lines):
                        node_code = '\n'.join(lines[start_line-1:end_line])
                    else:
                        node_code = ""
                    
                    child_node = CodeNode(
                        name=name,
                        node_type=node_type,
                        file_path=parent.file_path,
                        start_line=start_line,
                        end_line=end_line,
                        loc=self.metrics.calculate_loc(node_code),
                        complexity=self.metrics.calculate_complexity(node_code, language)
                    )
                    
                    # Track calls made by this node
                    calls = self._extract_calls(child, source_code, language)
                    child_node.call_count.outgoing = len(calls)
                    
                    # Store for call graph analysis
                    node_id = f"{parent.file_path}::{name}"
                    self.definitions[node_id] = child_node
                    self.call_references[node_id].extend(calls)
                    
                    parent.children.append(child_node)
                    
                    # Recursively extract children (methods within classes)
                    self._extract_hierarchy(child, source_code, language, child_node)
                else:
                    # Continue traversing even if this node isn't interesting
                    self._extract_hierarchy(child, source_code, language, parent)
            except Exception as e:
                logging.debug(f"Failed to process node in {parent.file_path}: {e}")
                continue

    def _calculate_call_counts(self, module: CodeNode) -> None:
        """Calculate incoming call counts based on call graph."""
        # Build reverse call map
        incoming_calls: Dict[str, int] = defaultdict(int)
        
        for caller_id, calls in self.call_references.items():
            for call_name in calls:
                # Try to match call to known definitions
                for def_id, def_node in self.definitions.items():
                    if def_node.name == call_name:
                        incoming_calls[def_id] += 1
        
        # Update call counts
        def update_counts(node: CodeNode, path_prefix: str = "") -> None:
            node_id = f"{node.file_path}::{node.name}" if path_prefix else node.file_path
            if node_id in incoming_calls:
                node.call_count.incoming = incoming_calls[node_id]
            
            for child in node.children:
                update_counts(child, f"{path_prefix}::{node.name}" if path_prefix else node.name)
        
        update_counts(module)

    def analyze_directory(self, exclude_patterns: Optional[List[str]] = None) -> CodeNode:
        """Analyze entire directory structure with progress tracking."""
        if exclude_patterns is None:
            exclude_patterns = [
                '*/node_modules/*', '*/.git/*', '*/venv/*', '*/__pycache__/*',
                '*/build/*', '*/dist/*', '*/target/*', '*/.pytest_cache/*',
                '*/coverage/*', '*/.coverage', '*/logs/*', '*/tmp/*', '*/temp/*'
            ]
        
        root_node = CodeNode(name=self.repo_path.name, node_type='root')
        
        # Find all source files
        source_files = []
        supported_extensions = [ext for ext in self.tree_sitter.SUPPORTED_LANGUAGES.keys() 
                               if self.tree_sitter.is_supported(ext)]
        
        if not supported_extensions:
            raise AnalysisError("No supported file types available for analysis")
        
        for extension in supported_extensions:
            pattern = f"**/*{extension}"
            files = list(self.repo_path.glob(pattern))
            
            # Filter out excluded patterns
            filtered_files = []
            for file_path in files:
                excluded = False
                path_str = str(file_path).replace('\\', '/')  # Normalize path separators
                
                for pattern in exclude_patterns:
                    # Convert glob pattern to regex-like matching
                    if '*/' in pattern:
                        pattern_parts = pattern.split('/')
                        if any(part.replace('*', '') in path_str for part in pattern_parts if part != '*'):
                            excluded = True
                            break
                    elif pattern in path_str:
                        excluded = True
                        break
                
                if not excluded:
                    filtered_files.append(file_path)
            
            source_files.extend(filtered_files)
        
        if not source_files:
            logging.warning("No source files found to analyze")
            return root_node
        
        logging.info(f"Found {len(source_files)} source files to analyze")
        
        # Analyze each file with progress tracking
        successful_analyses = 0
        failed_analyses = 0
        
        for i, file_path in enumerate(source_files):
            try:
                logging.info(f"Analyzing {file_path.relative_to(self.repo_path)} ({i+1}/{len(source_files)})")
                module_node = self.analyze_file(file_path)
                root_node.children.append(module_node)
                successful_analyses += 1
            except Exception as e:
                logging.error(f"Failed to analyze {file_path}: {e}")
                failed_analyses += 1
        
        logging.info(f"Analysis complete: {successful_analyses} successful, {failed_analyses} failed")
        
        # Calculate aggregate metrics
        try:
            self._calculate_aggregate_metrics(root_node)
        except Exception as e:
            logging.error(f"Failed to calculate aggregate metrics: {e}")
        
        return root_node

    def _calculate_aggregate_metrics(self, node: CodeNode) -> None:
        """Calculate aggregate metrics for parent nodes."""
        if not node.children:
            return
        
        total_loc = 0
        total_complexity = 0
        total_churn = 0
        total_incoming = 0
        total_outgoing = 0
        
        for child in node.children:
            self._calculate_aggregate_metrics(child)
            total_loc += child.loc
            total_complexity += child.complexity
            total_churn += child.churn
            total_incoming += child.call_count.incoming
            total_outgoing += child.call_count.outgoing
        
        if node.node_type in ['root', 'module'] and node.children:
            node.loc = total_loc
            node.complexity = total_complexity
            node.churn = total_churn
            node.call_count.incoming = total_incoming
            node.call_count.outgoing = total_outgoing


def main() -> None:
    """Main entry point for the code analyzer with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="Extract code metrics for visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/project
  %(prog)s /path/to/project -o my-analysis.json
  %(prog)s /path/to/project --exclude "*/tests/*" "*/docs/*"
        """
    )
    parser.add_argument("repo_path", help="Path to the code repository")
    parser.add_argument("-o", "--output", default="code-map.json", 
                       help="Output JSON file (default: code-map.json)")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--exclude", nargs="*", 
                       help="Additional patterns to exclude from analysis")
    parser.add_argument("--max-file-size", type=int, default=1024*1024,
                       help="Maximum file size to analyze in bytes (default: 1MB)")
    
    args = parser.parse_args()
    
    # Setup logging with appropriate format
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        repo_path = Path(args.repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {args.repo_path}")
        if not repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {args.repo_path}")
        
        output_path = Path(args.output)
        if output_path.exists() and not output_path.is_file():
            raise ValueError(f"Output path exists but is not a file: {args.output}")
        
        # Initialize analyzer
        logger.info(f"Initializing analyzer for {args.repo_path}")
        analyzer = CodeAnalyzer(args.repo_path)
        
        # Run analysis
        logger.info(f"Starting analysis of {args.repo_path}")
        exclude_patterns = args.exclude if args.exclude else None
        root_node = analyzer.analyze_directory(exclude_patterns=exclude_patterns)
        
        # Validate results
        if not root_node.children:
            logger.warning("No code modules found in analysis")
        
        # Convert to JSON format
        logger.info("Converting results to JSON format")
        output_data = root_node.to_dict()
        
        # Write output with backup
        if output_path.exists():
            backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
            output_path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis complete. Output written to {args.output}")
        
        # Print summary
        print(f"\n📊 Code Analysis Summary")
        print(f"Repository: {args.repo_path}")
        print(f"Total modules: {len(root_node.children)}")
        print(f"Total LOC: {root_node.loc:,}")
        print(f"Total complexity: {root_node.complexity}")
        print(f"Output file: {args.output}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Language breakdown
        if root_node.children:
            extensions = defaultdict(int)
            for child in root_node.children:
                if child.file_path:
                    ext = Path(child.file_path).suffix
                    extensions[ext] += 1
            
            print("\nFile type breakdown:")
            for ext, count in sorted(extensions.items()):
                print(f"  {ext}: {count} files")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
