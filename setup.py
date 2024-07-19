from setuptools import setup, find_packages

setup(
    name="sktorch",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A scikit-learn style wrapper for PyTorch models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sktorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.19.0",
    ],
)