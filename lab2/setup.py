from setuptools import setup, find_packages

setup(
    name="lab2",
    description="MLOps course - Lab 2",
    author="Dawid Stachowiak",
    author_email="dawid.stachowiak@student.put.poznan.pl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pytorch_lightning",
        "optuna",
        "neptune",
        "python-dotenv",
        "optuna-integration[pytorch_lightning]",
    ],
    python_requires=">=3.10",
)
