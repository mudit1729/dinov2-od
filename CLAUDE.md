# CLAUDE.md - DINOv2 Object Detection Project

## Commands
- Install dependencies: `pip install -r requirements.txt`
- Install in dev mode: `pip install -e .`
- Run training: `python train.py`
- Run single test: `pytest tests/test_file.py::test_function -v`
- Lint: `black . && isort . && flake8`
- Type check: `mypy .`

## Code Style Guidelines
- **Imports**: Group imports: stdlib, third-party, local. Use absolute imports.
- **Formatting**: Black with 88 char line length, follow PEP 8.
- **Types**: Use type hints for function parameters and return values.
- **Docstrings**: Google style docstrings with params and returns sections.
- **Naming**: snake_case for variables/functions, PascalCase for classes.
- **Model structure**: Keep backbone frozen with LoRA adapters active only.
- **Error handling**: Use explicit exception types with context information.
- **Training logging**: Track loss values and evaluation metrics per epoch.