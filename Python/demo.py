# import trythis as w
#
# in_img = "../images/image2.jpg"
#
# out_img = "../results"
#
# w.trythis(in_img, out_img)
import cv2
import os
from WBEmulator import WBEmulator as wb
wbColorAug = wb.WBEmulator()
in_img = "../images/image2.jpg"
filename, file_extension = os.path.splitext(in_img)
out_dir = "../results"
I = cv2.imread(in_img) # read the image

outNum = 5
outImgs, wb_pf = wbColorAug.generateWbsRGB(I,outNum)

for i in range(outNum):
    outImg = outImgs[:,:,:,i]
    cv2.imwrite(out_dir + '/' + os.path.basename(filename) + '_' + wb_pf[i] + file_extension, outImg * 255)
