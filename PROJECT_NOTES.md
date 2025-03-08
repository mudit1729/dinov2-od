# PROJECT_NOTES.md - DINOv2 Object Detection Project

## Commands
- Install dependencies: `python3 -m pip install -r requirements.txt`
- Install in dev mode: `python3 -m pip install -e .`
- Run single-GPU training: `python3 -m dino_detector.train`
- Run multi-GPU training: `python3 -m dino_detector.train --distributed`
- Run training with specific configurations:
  ```bash
  python3 -m dino_detector.train \
    --distributed \
    --world_size 2 \
    --train_images path/to/coco/train2017 \
    --train_annotations path/to/coco/annotations/instances_train2017.json \
    --val_images path/to/coco/val2017 \
    --val_annotations path/to/coco/annotations/instances_val2017.json \
    --output_dir outputs
  ```
- Run single test: `python3 -m pytest tests/test_file.py::test_function -v`
- Lint: `black . && isort . && flake8`
- Type check: `python3 -m mypy .`

## Code Style Guidelines
- **Imports**: Group imports: stdlib, third-party, local. Use absolute imports.
- **Formatting**: Black with 88 char line length, follow PEP 8.
- **Types**: Use type hints for function parameters and return values.
- **Docstrings**: Google style docstrings with params and returns sections.
- **Naming**: snake_case for variables/functions, PascalCase for classes.
- **Model structure**: Keep backbone frozen with LoRA adapters active only.
- **Error handling**: Use explicit exception types with context information.
- **Training logging**: Track loss values and evaluation metrics per epoch.