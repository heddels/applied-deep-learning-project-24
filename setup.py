# This file is used to turn the scr as local package
from setuptools import setup, find_packages

setup(
    name="media_bias_detection",  # Name of your package
    version="0.1",  # Version number
    description="Multi-Task Learning Model for Media Bias Detection",
    author="Hedda Fiedler",
    packages=find_packages(where="src"),  # Tells pip to look in src directory
    package_dir={"": "src"},  # Root directory for the package
    install_requires=[  # Core Deep Learning
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "torchmetrics>=1.0.0",

        # Data Processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "statistics>=1.0.0",

        # Tokenization
        "tokenizers>=0.13.0",  # Added - needed for DistilBertTokenizerFast

        # Experiment Tracking
        "wandb>=0.15.0",
        "tqdm>=4.65.0",

        # System & Memory
        "psutil>=5.9.0",

        # Configuration & Utils
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "python-box>=7.0.1",
        "ensure>=1.0.3",
        "joblib>=1.3.0",
        "pathlib>=1.0.1",
    ],
    python_requires=">=3.8",  # Minimum Python version
)
