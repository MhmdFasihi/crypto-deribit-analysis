from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anomaly_option",
    version="0.1.0",
    author="Mohammad Fasihi",
    author_email="your.email@example.com",
    description="A comprehensive system for analyzing cryptocurrency price volatility and options data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MhmdFasihi/crypto-deribit-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "tqdm>=4.66.0",
        "dash>=2.14.0",
        "dash-bootstrap-components>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anomaly-option=anomaly_option.core.analysis_system:main",
        ],
    },
) 