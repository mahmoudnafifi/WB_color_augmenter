import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WBcolorAug-afifi",
    version="0.0.0",
    author="Mahmoud Afifi",
    author_email="mafifi@eecs.yorku.ca",
    description="Image color augmentation by emulation white balance effects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahmoudnafifi/WB_color_augmenter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)