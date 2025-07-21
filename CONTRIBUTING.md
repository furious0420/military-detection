# Contributing to YOLO Multi-Class Detection

Thank you for your interest in contributing to this project! 🎉

## 🚀 How to Contribute

### 1. Fork the Repository
- Click the "Fork" button at the top right of the repository page
- Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/yolo-multiclass-detection.git
cd yolo-multiclass-detection
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 pre-commit
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
```

### 4. Make Your Changes
- Write clean, readable code
- Add comments and docstrings
- Follow Python PEP 8 style guidelines
- Add tests for new features

### 5. Test Your Changes
```bash
# Run tests
python -m pytest

# Check code style
black --check .
flake8 .

# Test the main functionality
python universal_detector.py "test_image.jpg"
```

### 6. Commit and Push
```bash
git add .
git commit -m "Add: your descriptive commit message"
git push origin feature/your-feature-name
```

### 7. Create Pull Request
- Go to your fork on GitHub
- Click "New Pull Request"
- Provide a clear description of your changes

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove)
- Keep the first line under 50 characters
- Add detailed description if needed

### Testing
- Add tests for new features
- Ensure all existing tests pass
- Test with different image types and sizes
- Verify performance doesn't degrade

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Error messages (full traceback)
- Steps to reproduce
- Sample images (if applicable)

## 💡 Feature Requests

For new features, please:
- Check if it already exists in issues
- Describe the use case
- Explain why it would be valuable
- Consider implementation complexity

## 🔧 Development Areas

We welcome contributions in these areas:

### Model Improvements
- New class additions
- Performance optimizations
- Accuracy improvements
- Model compression techniques

### Features
- Real-time video processing
- Batch processing utilities
- Export format support
- Visualization enhancements

### Documentation
- API documentation
- Tutorials and examples
- Performance benchmarks
- Deployment guides

### Testing
- Unit tests
- Integration tests
- Performance tests
- Cross-platform testing

## 📞 Questions?

- Open an issue for questions
- Check existing issues first
- Be respectful and constructive

## 🙏 Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Given credit for their contributions

Thank you for helping make this project better! 🌟
