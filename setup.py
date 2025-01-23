from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="L-SARP",
    version="0.1.0",
    description="Language-based Scene Aware Robot Planning",
    author="Zeno Hamers",  # Add your name
    author_email="zeno.hamers@gmail.com",  # Add your email
    packages=find_packages(where="source"),  # Look for packages in source directory
    package_dir={"": "source"},  # Tell setuptools where to find the packages
    include_package_data=True,
    python_requires=">=3.8",  # Based on requirements like torch 2.0.1
    install_requires=required,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            # Add any CLI entry points here if needed
            # "my-command=module.path:main_function",
        ],
    },
    project_urls={
        # Add relevant URLs like:
        # "Bug Tracker": "https://github.com/username/project/issues",
        # "Documentation": "https://project.readthedocs.io/",
        # "Source Code": "https://github.com/username/project",
    },
)
