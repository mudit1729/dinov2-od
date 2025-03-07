from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="dino_detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Object detection using DINOv2 backbone with LoRA adapters and DETR decoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dinov2-od",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)