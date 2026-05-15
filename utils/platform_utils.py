"""
Platform-specific utilities for cross-platform compatibility.

This module provides utilities for handling platform-specific operations
such as DLL loading, path handling, and environment configuration.
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Optional


def get_conda_dll_paths() -> List[str]:
    """
    Get potential conda/anaconda DLL paths for the current environment.
    
    Returns cross-platform compatible paths for library binaries.
    Works on Windows, Linux, and macOS.
    
    Returns:
        List[str]: List of potential DLL/library paths
    """
    paths = []
    
    if os.name == 'nt':  # Windows
        # Try to get from environment variables first
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            paths.append(os.path.join(conda_prefix, 'Library', 'bin'))
        
        # Add Python prefix (works for venv and conda)
        if hasattr(sys, 'prefix'):
            paths.append(os.path.join(sys.prefix, 'Library', 'bin'))
        
        # Add common conda installation paths (as fallback)
        home_dir = os.path.expanduser('~')
        common_paths = [
            os.path.join(home_dir, 'anaconda3', 'Library', 'bin'),
            os.path.join(home_dir, 'miniconda3', 'Library', 'bin'),
            os.path.join(home_dir, '.conda', 'envs', '*', 'Library', 'bin'),
            'C:\\ProgramData\\Anaconda3\\Library\\bin',
            'C:\\ProgramData\\Miniconda3\\Library\\bin',
        ]
        paths.extend(common_paths)
    
    elif platform.system() == 'Linux':
        # Linux library paths
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            paths.append(os.path.join(conda_prefix, 'lib'))
        
        if hasattr(sys, 'prefix'):
            paths.append(os.path.join(sys.prefix, 'lib'))
    
    elif platform.system() == 'Darwin':  # macOS
        # macOS library paths
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            paths.append(os.path.join(conda_prefix, 'lib'))
        
        if hasattr(sys, 'prefix'):
            paths.append(os.path.join(sys.prefix, 'lib'))
    
    return paths


def setup_dll_directories() -> bool:
    """
    Setup DLL directories for Windows or library paths for other platforms.
    
    This function attempts to add necessary library directories to the
    system path to enable loading of required DLLs (especially for PyTorch
    and TensorFlow on Windows).
    
    Returns:
        bool: True if at least one valid path was added, False otherwise
    """
    if os.name == 'nt':  # Windows
        try:
            paths = get_conda_dll_paths()
            
            for path in paths:
                if os.path.exists(path):
                    try:
                        os.add_dll_directory(path)
                        return True
                    except (OSError, AttributeError):
                        # add_dll_directory might not be available in older Python
                        # or path might not be valid
                        continue
            
            return False
        
        except Exception:
            return False
    
    else:
        # On Linux/macOS, LD_LIBRARY_PATH/DYLD_LIBRARY_PATH are typically managed
        # by the conda/venv activation scripts
        return True


def get_platform_cache_dir() -> Path:
    """
    Get platform-appropriate cache directory.
    
    Returns:
        Path: Cache directory path
    """
    if os.name == 'nt':  # Windows
        cache_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return Path(cache_dir) / 'cv_training_pipeline' / 'cache'
    
    elif platform.system() == 'Darwin':  # macOS
        return Path.home() / 'Library' / 'Caches' / 'cv_training_pipeline'
    
    else:  # Linux and others
        xdg_cache = os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))
        return Path(xdg_cache) / 'cv_training_pipeline'


def get_platform_data_dir() -> Path:
    """
    Get platform-appropriate data directory.
    
    Returns:
        Path: Data directory path
    """
    if os.name == 'nt':  # Windows
        data_dir = os.environ.get('APPDATA', os.path.expanduser('~'))
        return Path(data_dir) / 'cv_training_pipeline'
    
    elif platform.system() == 'Darwin':  # macOS  
        return Path.home() / 'Library' / 'Application Support' / 'cv_training_pipeline'
    
    else:  # Linux and others
        xdg_data = os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local' / 'share'))
        return Path(xdg_data) / 'cv_training_pipeline'


def normalize_path(path: str) -> str:
    """
    Normalize path for current platform.
    
    Converts forward/backward slashes as needed and expands user directory.
    
    Args:
        path: Input path string
        
    Returns:
        str: Normalized path
    """
    return str(Path(path).expanduser().resolve())


# Initialize DLL directories on module import (for Windows)
_dll_setup_success = setup_dll_directories()


__all__ = [
    'get_conda_dll_paths',
    'setup_dll_directories',
    'get_platform_cache_dir',
    'get_platform_data_dir',
    'normalize_path',
]
