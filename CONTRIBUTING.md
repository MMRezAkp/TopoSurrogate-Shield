# Contributing to Neural Network Activation Extractor

Thank you for considering contributing to this project! This document outlines the process for contributing and helps ensure consistency across contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment for all contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your feature or bugfix
5. Make your changes
6. Test your changes
7. Submit a pull request

## How to Contribute

### Types of Contributions

- **Bug Reports**: Report bugs using GitHub issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes, new features, or improvements
- **Documentation**: Improve or add documentation
- **Examples**: Add new example scripts or improve existing ones

### What We're Looking For

- Support for additional neural network architectures
- Performance optimizations
- New topological analysis methods
- Better visualization tools
- Improved documentation and examples
- Bug fixes and error handling improvements

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/neural-activation-extractor.git
   cd neural-activation-extractor
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Install in development mode**:
   ```bash
   pip install -e .
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions and classes focused and single-purpose
- Use type hints where appropriate

### Code Organization

- Place new architectures in `model.py`
- Add new tap modes in the `ImprovedActivationExtractor` class
- Keep utility functions separate and well-documented
- Follow the existing module structure

### Documentation

- Add docstrings for all public functions and classes
- Update README.md for new features
- Add examples for new functionality
- Keep comments concise and helpful

## Testing

### Running Tests

```bash
# Run basic functionality tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_activation_extractor.py

# Run with coverage
python -m pytest --cov=activation_topotroj tests/
```

### Writing Tests

- Write tests for new features and bug fixes
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies when appropriate

### Test Structure

```python
def test_function_name():
    """Test description."""
    # Arrange
    # Act
    # Assert
```

## Pull Request Process

### Before Submitting

1. **Update documentation**: Ensure README, docstrings, and examples are updated
2. **Add tests**: Include tests for new functionality
3. **Run tests**: Ensure all tests pass
4. **Check code style**: Follow coding standards
5. **Update changelog**: Add entry to CHANGELOG.md

### Pull Request Guidelines

1. **Create a descriptive title**: Clearly describe what the PR does
2. **Add a detailed description**: Explain the changes and why they're needed
3. **Reference issues**: Link to related issues using "Fixes #123" or "Closes #123"
4. **Keep changes focused**: One feature or fix per PR
5. **Update documentation**: Include relevant documentation updates

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes don't break existing functionality
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment details**: OS, Python version, PyTorch version
- **Steps to reproduce**: Clear, minimal steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Complete error messages and stack traces
- **Sample code**: Minimal code example that reproduces the issue

### Feature Requests

When requesting features, please include:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: Your idea for how to solve it
- **Alternatives considered**: Other approaches you've considered
- **Additional context**: Any other relevant information

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## Architecture-Specific Contributions

### Adding New Architectures

When adding support for a new architecture:

1. **Update `model.py`**: Add the new architecture to the `get_model()` function
2. **Update documentation**: Add the architecture to supported lists
3. **Add examples**: Include usage examples for the new architecture
4. **Test compatibility**: Ensure all tap modes work (where applicable)
5. **Update architecture detection**: Modify the auto-detection logic if needed

### Adding New Tap Modes

When adding new tap modes:

1. **Follow existing patterns**: Use similar structure to existing modes
2. **Add comprehensive documentation**: Explain when and why to use the mode
3. **Test with multiple architectures**: Ensure compatibility
4. **Add examples**: Include usage examples
5. **Update help text**: Add to command-line help

## Getting Help

- **GitHub Issues**: For questions about usage or potential bugs
- **GitHub Discussions**: For general questions and community discussions
- **Documentation**: Check the README and examples first

## Recognition

Contributors will be recognized in:
- The contributors section of the README
- The GitHub contributors page
- Release notes for significant contributions

Thank you for contributing to make this project better!
