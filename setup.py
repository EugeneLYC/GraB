import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="orderedsampler",
    version="0.0.1",
    author="Yucheng Lu",
    author_email="yl2967@cornell.edu",
    description="pytorch-based OrderedSampler that supports example ordering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EugeneLYC/orderedsampler",
    project_urls={
        "Bug Tracker": "https://github.com/EugeneLYC/orderedsampler",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)