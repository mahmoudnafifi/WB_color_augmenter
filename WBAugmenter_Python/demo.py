################################################################################
# Copyright (c) 2019-present, Mahmoud Afifi
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#
# Please, cite the following paper if you use this code:
# Mahmoud Afifi and Michael S. Brown. What else can fool deep learning?
# Addressing color constancy errors on deep neural network performance. ICCV,
# 2019
#
# Email: mafifi@eecs.yorku.ca | m.3afifi@gmail.com
################################################################################

from PIL import Image
import os
from WBAugmenter import WBEmulator as wbAug

wbColorAug = wbAug.WBEmulator()  # create an instance of the WB emulator
in_img = "../images/image1.jpg"  # input image filename
filename, file_extension = os.path.splitext(in_img)  # get file parts
out_dir = "../results"  # output directory
os.makedirs(out_dir, exist_ok=True)
I = Image.open(in_img)  # read the image
outNum = 10  # number of images to generate (should be <= 10)
# generate new images with different WB settings
outImgs, wb_pf = wbColorAug.generateWbsRGB(I, outNum)
for i in range(outNum):  # save images
    outImg = outImgs[i]  # get the ith output image
    outImg.save(out_dir + '/' + os.path.basename(filename) +
                '_' + wb_pf[i] + file_extension)  # save it



