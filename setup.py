"""Setup configuration for Dinnovos Agent"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version
version = {}
with open("dinnovos/version.py", "r") as f:
    exec(f.read(), version)

setup(
    name="dinnovos-agent",
    version=version["__version__"],
    author="Dinnovos",
    author_email="developer.dinnovos@gmail.com",
    description="Dinnovos Agent - Agile AI Agents with multi-LLM support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dinnovos/dinnovos-agent",
    project_urls={
        "Bug Tracker": "https://github.com/dinnovos/dinnovos-agent/issues",
        "Documentation": "https://github.com/dinnovos/dinnovos-agent/docs",
        "Source Code": "https://github.com/dinnovos/dinnovos-agent",
    },
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (none required for base functionality)
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "google": ["google-generativeai>=0.3.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "google-generativeai>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="ai agents llm openai anthropic google gemini claude gpt dinnovos",
    include_package_data=True,
)