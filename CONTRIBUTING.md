# Contributing to `python-oephys`

Thank you for your interest in contributing to `python-oephys`! We welcome contributions from the community, whether they are bug reports, feature requests, documentation improvements, or new code.

## 🐛 Reporting Bugs

If you find a bug, please open an issue on our [GitHub Issues](https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/issues) page. Please include:
- A clear and descriptive title.
- Steps to reproduce the bug.
- Actual vs. expected behavior.
- Your OS and Python version.

## ✨ Feature Requests

We’re always looking for ways to improve! If you have an idea for a new feature, please open an issue and describe:
- What problem it solves.
- How it should work (ideally with a code snippet or pseudocode).

## 🔧 Pull Request Guidelines

1. **Fork the repository** and create your branch from `main`.
2. **Follow the style**: Use PEP 8 for Python code and provide clear docstrings.
3. **Add tests**: If you're adding code, please add corresponding tests in the `tests/` directory.
4. **Update documentation**: If your change affects the API or CLI, update the docstrings and README.
5. **Run tests**: Ensure all tests pass before submitting your PR:
   ```bash
   pytest
   ```

## 🧪 Development Setup

1. Clone the repo: `git clone https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys.git`
2. Install with development dependencies:
   ```bash
   pip install -e ".[gui,ml,docs]"
   pip install pytest pytest-cov
   ```

## 📜 Code of Conduct

Please be respectful and professional in all interactions within this project. We aim to foster a welcoming and inclusive environment for everyone.

---
Made with ❤️ by the Neuromechatronics Lab
