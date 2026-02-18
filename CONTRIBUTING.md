# Contributing to AI Job Displacement Analysis

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct
- Treat all contributors with respect
- Provide constructive feedback
- Focus on the code, not the person
- Be inclusive and welcoming

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- pip or conda

### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/Muhammad-Farooq-13/ai-job-displacement-analysis.git
cd ai-job-displacement-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bugfixes:
git checkout -b bugfix/issue-description
```

### 2. Make Your Changes
- Write clean, documented code
- Follow PEP 8 style guide
- Add type hints to functions
- Include docstrings for modules and functions

### 3. Test Your Changes
```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### 4. Commit Your Changes
```bash
# Use conventional commit format
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in preprocessing"
git commit -m "docs: update README with new examples"
git commit -m "test: add unit tests for feature"
git commit -m "refactor: improve code organization"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```
Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Link to related issues if applicable
- Screenshots for UI changes

## Code Standards

### Type Hints
```python
from typing import List, Dict, Optional, Tuple

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Process data and return cleaned DataFrame and metadata."""
    pass
```

### Docstrings (Google Style)
```python
def calculate_risk_score(features: np.ndarray) -> float:
    """Calculate AI displacement risk score.
    
    Args:
        features: Array of job characteristics (shape: n_features,)
    
    Returns:
        Risk score between 0 and 1
    
    Raises:
        ValueError: If features array is empty
    
    Example:
        >>> features = np.array([0.5, 0.3, 0.7])
        >>> score = calculate_risk_score(features)
    """
    pass
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Data loading started")
logger.warning("Missing values detected in column X")
logger.error("Failed to load model: {error}")
```

## Testing

### Write Tests for New Features
```python
# tests/test_new_feature.py
import pytest
from src.new_module import new_function

class TestNewFunction:
    def test_basic_functionality(self):
        result = new_function(input_data)
        assert result == expected_output
    
    def test_error_handling(self):
        with pytest.raises(ValueError):
            new_function(invalid_input)
```

### Test Coverage
- Aim for >80% code coverage
- Test edge cases and error conditions
- Test with various input types

## Documentation

### Update Documentation
- Update README for user-facing changes
- Add docstrings to new functions
- Update API documentation if endpoints change
- Add examples for complex features

### Example Format
```markdown
## New Feature

### Usage
```python
from src.module import function
result = function(parameter)
```

### Parameters
- `parameter`: Description

### Returns
- Description of return value
```

## Review Process

1. **Automated Checks**
   - Tests must pass
   - Code coverage >80%
   - No linting errors
   - Type checking passes

2. **Code Review**
   - At least one approval required
   - Address feedback promptly
   - Update based on suggestions

3. **Merge**
   - Maintainer will merge after approval
   - Delete feature branch after merge

## Issue Reporting

### Bug Reports
Include:
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error traceback
- Minimal reproducible example

### Feature Requests
Include:
- Use case and motivation
- Proposed solution
- Alternative approaches
- Example code if applicable

## Project Structure
```
src/               # Core modules
├── config.py      # Configuration
├── data_loader.py # Data ingestion
├── preprocessing.py
├── feature_engineering.py
├── model_training.py
├── evaluation.py
└── utils.py

tests/             # Test files
notebooks/         # Analysis notebooks
deployment/        # Deployment configs
models/            # Model artifacts
```

## Adding Dependencies

1. Update `requirements.txt` with version pinning
2. Add to `setup.py` if part of core package
3. Document why dependency is needed
4. Test compatibility with Python 3.9+

## Release Process

1. Create release branch: `git checkout -b release/v1.1.0`
2. Update version in `src/__init__.py`
3. Update CHANGELOG
4. Create PR and merge to main
5. Tag release: `git tag v1.1.0`
6. Push tags: `git push origin v1.1.0`

## Questions?
- Check existing issues and discussions
- Open a new issue with `question` label
- Review documentation in `/docs`

## Recognition
Contributors will be recognized in:
- README contributors section
- Release notes
- GitHub contributors graph

Thank you for contributing to making this project better! 🎉
