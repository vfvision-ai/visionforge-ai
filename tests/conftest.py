"""
Conftest for pytest - shared fixtures and configuration
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import os


@pytest.fixture(scope='session')
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope='session')
def test_data_dir(temp_dir):
    """Create a test data directory"""
    data_dir = temp_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope='session')
def test_upload_dir(temp_dir):
    """Create a test upload directory"""
    upload_dir = temp_dir / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@pytest.fixture(scope='session')
def test_experiments_dir(temp_dir):
    """Create a test experiments directory"""
    exp_dir = temp_dir / 'experiments'
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set environment variables for testing"""
    monkeypatch.setenv('ENVIRONMENT', 'development')
    monkeypatch.setenv('DEBUG', 'true')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('METRICS_ENABLED', 'false')


@pytest.fixture
def sample_image_file(temp_dir):
    """Create a sample image file for testing"""
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    img_path = temp_dir / 'test_image.jpg'
    img.save(img_path)
    
    return img_path
