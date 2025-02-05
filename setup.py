from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="sentiment_api",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "sentiment-api=src.main:run_app",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
