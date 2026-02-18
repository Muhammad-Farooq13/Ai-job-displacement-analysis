"""
Setup configuration for AI Job Displacement Analysis project.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ai-job-displacement-analysis",
    version="1.0.0",
    author="Muhammad Farooq",
    author_email="mfarooqshafee333@gmail.com",
    description="ML-powered analysis of AI-driven job displacement trends 2020-2026 | GradientBoosting model (R²=0.93) | Production-ready API, Docker, CI/CD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Muhammad-Farooq-13/ai-job-displacement-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/Muhammad-Farooq-13/ai-job-displacement-analysis/issues",
        "Documentation": "https://github.com/Muhammad-Farooq-13/ai-job-displacement-analysis/blob/main/README.md",
    },
    packages=find_packages(exclude=["tests", "notebooks", "data"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.2.2",
        "xgboost>=1.7.5",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "jupyter>=1.0.0",
        "scipy>=1.10.1",
        "plotly>=5.13.0",
        "shap>=0.42.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "isort>=5.12.0",
        ],
        "deployment": [
            "flask>=2.3.1",
            "streamlit>=1.22.1",
            "gunicorn>=20.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-jobs=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
