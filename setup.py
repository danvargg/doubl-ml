from setuptools import setup, find_packages

setup(
    name="fitml",
    version="0.1.0",
    description="Lightweight ML pipeline for clothing fit prediction",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourusername/fitml",
    packages=find_packages(include=["fitml", "fitml.*"]),
    install_requires=[
        "numpy>=1.21",
        "tensorflow>=2.9",
        "mediapipe>=0.8"
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "fitml-train=fitml.train:main"
        ]
    }
)
