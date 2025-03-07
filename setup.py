from setuptools import setup, find_packages

setup(
    name="ai-co-scientist",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gemini-api-client",
        "langchain",
        "numpy",
        "pandas",
        "requests",
        "pydantic",
        "asyncio",
        "aiohttp",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "mypy",
            "ruff",
        ]
    },
    python_requires=">=3.9",
)