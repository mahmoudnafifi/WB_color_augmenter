#####################################################################################
# Copyright (c) 2019-present, Mahmoud Afifi
#
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.
#
# Please, cite the following paper if you use this code:
# Mahmoud Afifi and Michael S. Brown. What else can fool deep learning? Addressing
# color constancy errors on deep neural network performance. ICCV, 2019
#
# Email: mafifi@eecs.yorku.ca | m.3afifi@gmail.com
######################################################################################

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