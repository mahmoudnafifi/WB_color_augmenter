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

import cv2
import os
from WBAugmenter import WBEmulator as wbAug
wbColorAug = wbAug.WBEmulator() # create an instance of the WB emulator
in_img = "../images/image2.jpg" # input image filename
filename, file_extension = os.path.splitext(in_img) # get file parts
out_dir = "../results" # output directory
os.makedirs(out_dir, exist_ok=True)
I = cv2.imread(in_img) # read the image
outNum = 5 # number of images to generate (should be <= 10)
outImgs, wb_pf = wbColorAug.generateWbsRGB(I,outNum) # generate new images with different WB settings
for i in range(outNum): # save images
    outImg = outImgs[:,:,:,i] # get the ith output image
    cv2.imwrite(out_dir + '/' + os.path.basename(filename) +
                '_' + wb_pf[i] + file_extension, outImg * 255) # save it



