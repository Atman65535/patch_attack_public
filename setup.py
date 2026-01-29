from setuptools import setup, find_packages

setup(
    name="D4A",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "segmentation-models-pytorch",
        "diffusers",
        "transformers",
    ],
    python_requires=">=3.8",
)