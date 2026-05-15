# Contributing to VisionForge

First off, thank you for considering contributing to this project! It's people like you that make this tool better for everyone.

## 🌟 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce the behavior
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**:
  - OS: [e.g., Ubuntu 22.04, Windows 11]
  - Python Version: [e.g., 3.9.10]
  - Framework: [e.g., TensorFlow 2.15.0]
  - GPU (if applicable): [e.g., NVIDIA RTX 3090]
- **Logs**: Relevant error messages or logs
- **Screenshots**: If applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to most users
- **List any alternative solutions** you've considered

### Pull Requests

1. **Fork the repository** and create your branch from `develop`
2. **Follow the code style** (use Black formatter, isort for imports)
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** (`pytest tests/`)
6. **Write meaningful commit messages**

## 🔧 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## 📝 Code Style Guidelines

### Python Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 88)
- **isort** for import sorting
- **mypy** for type checking
- **pylint** for linting

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy core/ utils/

# Run linting
pylint core/ utils/
```

### Coding Standards

1. **Type Hints**: Use type hints for all function signatures
```python
def analyze_dataset(path: str, task_type: str = "auto") -> DatasetInfo:
    ...
```

2. **Docstrings**: Use Google-style docstrings
```python
def train_model(config: Dict[str, Any]) -> ModelResults:
    """
    Train a model with the given configuration.
    
    Args:
        config: Training configuration dictionary containing:
            - dataset_path: Path to dataset
            - model_architecture: Model architecture name
            - epochs: Number of training epochs
    
    Returns:
        ModelResults object with training metrics and model path
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
    """
    ...
```

3. **Error Handling**: Use specific exceptions and provide helpful error messages
```python
if not dataset_path.exists():
    raise ValueError(f"Dataset path not found: {dataset_path}")
```

4. **Logging**: Use the logger instead of print statements
```python
from utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Starting training...")
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Use pytest for all tests
- Aim for >80% code coverage
- Test both success and failure cases

```python
import pytest
from core.dataset_analyzer import DatasetAnalyzer

def test_analyze_valid_dataset():
    """Test dataset analysis with valid input."""
    analyzer = DatasetAnalyzer()
    result = analyzer.analyze("path/to/dataset")
    assert result.num_classes > 0

def test_analyze_invalid_path():
    """Test dataset analysis with invalid path."""
    analyzer = DatasetAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze("invalid/path")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_dataset_analyzer.py

# Run tests matching pattern
pytest -k "test_analyze"
```

## 🌿 Git Workflow

### Branch Naming

- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `fix/bug-description`
- **Documentation**: `docs/what-you-updated`
- **Refactoring**: `refactor/what-you-refactored`

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(dataset): add support for COCO format detection

Implements automatic detection of COCO annotation format
with validation of required fields.

Closes #123
```

```
fix(trainer): handle empty validation set gracefully

Previously crashed when validation set was empty.
Now shows warning and skips validation.
```

## 🔍 Code Review Process

1. **Self-review** your code before submitting
2. **Ensure CI passes** (all tests, linting, formatting)
3. **Respond to feedback** constructively
4. **Keep PRs focused** - one feature/fix per PR
5. **Update your PR** based on review comments

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows the style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass locally
- [ ] No new warnings introduced
- [ ] Commit messages are clear
- [ ] PR description explains the changes

## 📚 Documentation

### Documentation Standards

- Update README.md for user-facing changes
- Add docstrings to all public functions/classes
- Update API documentation for new features
- Include code examples where helpful
- Keep documentation concise and clear

### Building Documentation

```bash
# Generate API documentation (if using Sphinx)
cd docs/
make html
```

## 🏗️ Project Structure

```
visionforge/
├── core/              # Core ML functionality
│   ├── dataset_analyzer.py
│   ├── model_selector.py
│   ├── trainer.py
│   └── optimizer.py
├── utils/             # Utility modules
│   ├── config.py
│   ├── logger.py
│   ├── metrics.py
│   └── callbacks.py
├── tests/             # Test suite
│   ├── test_dataset_analyzer.py
│   └── test_trainer.py
├── app.py             # Streamlit web interface
└── requirements.txt   # Dependencies
```

## 🤝 Community Guidelines

- **Be respectful** and considerate
- **Be patient** with new contributors
- **Give constructive feedback**
- **Ask questions** when unclear
- **Help others** when you can

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private concerns

## 🎉 Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- GitHub contributors page

Thank you for contributing! 🙌
