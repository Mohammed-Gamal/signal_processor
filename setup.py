from setuptools import setup, find_packages

setup(
    name="signal_processor",
    version="0.1.0",
    author="Mohamed Gamal",
    author_email="1Mohamed.Gamal54@gmail.com",
    description="A package for generating and processing discrete-time signals",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mohammed-Gamal/signal_processor",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
