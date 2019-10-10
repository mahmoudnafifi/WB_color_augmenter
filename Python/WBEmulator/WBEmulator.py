import numpy as np
import numpy.matlib
import cv2
import random as rnd
from scipy.spatial.distance import cdist


class WBEmulator:
    def __init__(self):
        self.features = np.load('features.npy') # training encoded features
        self.mappingFuncs = np.load('./mappingFuncs.npy') # mapping functions to emulate WB effects
        self.encoderWeights = np.load('./encoderWeights.npy') # weight matrix for histogram encoding
        self.encoderBias = np.load('./encoderBias.npy') # bias vector for histogram encoding
        self.h = 60 # histogram bin width
        self.K = 25 # K value for nearest neighbor searching
        self.sigma = 0.25 # fall off factor for KNN
        self.wb_photo_finishing =  ['_F_AS', '_F_CS', '_S_AS', '_S_CS',
                                    '_T_AS', '_T_CS', '_C_AS', '_C_CS',
                                    '_D_AS', '_D_CS'] # WB & photo finishing styles

    def encode(self, hist): #encode given histogram
        histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                       (1, int(hist.size / 3)), order="F")  # reshaped red layer of histogram
        histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                    (1, int(hist.size / 3)), order="F")  # reshaped green layer of histogram
        histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                    (1, int(hist.size / 3)), order="F")  # reshaped blue layer of histogram
        hist_reshaped = np.append(histR_reshaped,
                                  [histG_reshaped, histB_reshaped])  # reshaped histogram n * 3 (n = h*h)
        feature = np.dot(hist_reshaped - self.encoderBias.transpose(), self.encoderWeights)  # compute compacted histogram feature
        return feature

    def rgbUVhist(self, I): #compute the RGB-uv histogram tensor
        #I = im2double(I) # convert to double
        # compute RGB-uv histogram
        sz = np.shape(I)  # get size of current image
        if sz[0] * sz[1] > 202500:  # if it is larger than 450*450
            factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
            newH = int(np.floor(sz[0] * factor))
            newW = int(np.floor(sz[1] * factor))
            I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)  # resize image
        II = I.reshape(int(I.size / 3), 3)  # n*3
        inds = np.where((II[:, 0] > 0) & (II[:, 1] > 0) & (II[:, 2] > 0))  # remove any zero pixels
        R = II[inds, 0]  # red channel
        G = II[inds, 1]  # green channel
        B = II[inds, 2]  # blue channel
        I_reshaped = np.concatenate((R, G, B), axis=0).transpose()  # reshaped image (wo zero values)
        eps = 6.4 / self.h
        A = np.arange(-3.2, 3.19, eps)  # dummy vector
        hist = np.zeros((A.size, A.size, 3))  # histogram will be stored here
        Iy = np.sqrt(np.power(I_reshaped[:, 0], 2) + np.power(I_reshaped[:, 1], 2) +
                     np.power(I_reshaped[:, 2], 2))  # intensity vector
        for i in range(3):  # for each histogram layer, do
            r = []  # excluded channels will be stored here
            for j in range(3):  # for each color channel do
                if j != i:  # if current color channel does not match current histogram layer,
                    r.append(j)  # exclude it
            Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])  # current color channel / the first excluded channel
            Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])  # current color channel / the second excluded channel
            diff_u = np.abs(np.matlib.repmat(Iu, np.size(A), 1).transpose() - np.matlib.repmat(A, np.size(Iu),
                                                                                               1))  # differences in u space
            diff_v = np.abs(np.matlib.repmat(Iv, np.size(A), 1).transpose() - np.matlib.repmat(A, np.size(Iv),
                                                                                               1))  # differences in v space
            diff_u[diff_u >= (eps / 2)] = 0  # do not count any pixel has difference beyond the threshold in the u space
            diff_u[diff_u != 0] = 1  # remaining pixels will be counted
            diff_v[diff_v >= (eps / 2)] = 0  # do not count any pixel has difference beyond the threshold in the v space
            diff_v[diff_v != 0] = 1  # remaining pixels will be counted
            temp = (np.matlib.repmat(Iy, np.size(A), 1) * (diff_u).transpose())  # Iy .* diff_u' (.* element-wise mult)
            hist[:, :, i] = np.dot(temp, diff_v)  # initialize current histogram layer with Iy .* diff' * diff_v
            norm_ = np.sum(hist[:, :, i], axis=None)  # compute sum of hist for normalization
            hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
        return hist


    def generateWbsRGB(self, I, outNum = 10): #emulate WB | outNum number of images to generate is otpional
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
        I = im2double(I)  # convert to double
        feature = self.encode(self.rgbUVhist(I))
        if outNum < len(self.wb_photo_finishing):  # if selected outNum is less than number of available WB & photo finishing (PF) styles,
            #inds = np.random.permutation(len(self.wb_photo_finishing))  # randomize and select outNum WB & PF styles
            #wb_pf = self.wb_photo_finishing[inds[0:outNum - 1]]  # wb_pf now represents the selected WB & PF styles
            wb_pf = rnd.sample(self.wb_photo_finishing, outNum)
            inds = []
            for j in range(outNum):
                inds.append(self.wb_photo_finishing.index(wb_pf[j]))

        else:  # if the selected number of images equals the available WB & PF styles,
            wb_pf = self.wb_photo_finishing  # then wb_pf = wb_photo_finishing
            inds = list(range(0, len(wb_pf)))  # inds is simply from 0 to the number of available WB & PF styles
        synthWBimages = np.zeros((I.shape[0], I.shape[1],
                                  I.shape[2], len(wb_pf)))  # synthetic images will be stored here

        D_sq = np.einsum('ij, ij ->i', self.features, self.features)[:, None] + \
               np.einsum('ij, ij ->i', feature, feature) - \
               2 * self.features.dot(feature.T)  # squared euclidean distances

        idH = D_sq.argpartition(self.K, axis=0)[:self.K]  # get smallest K distances
        dH = np.sqrt(
            np.take_along_axis(D_sq, idH, axis=0))  # square root nearest distances to get real euclidean distances
        sorted_idx = dH.argsort(axis=0)  # get sorting indices
        idH = np.take_along_axis(idH, sorted_idx, axis=0)  # sort distance indices
        dH = np.take_along_axis(dH, sorted_idx, axis=0)  # sort distances
        weightsH = np.exp(-(np.power(dH, 2)) /
                          (2 * np.power(self.sigma, 2)))  # compute blending weights
        weightsH = weightsH / sum(weightsH)  # normalize blending weights
        for i in range(len(inds)):  # for each of the retried training examples, do
            ind = inds[i]  # for each WB & PF style,
            mf = sum(np.reshape(np.matlib.repmat(weightsH, 1, 27), (25, 1, 9, 3)) *
                     self.mappingFuncs[(idH - 1) * 10 + ind, :])  # compute the mapping function
            mf = mf.reshape(9, 3, order="F")  # reshape it to be 9 * 3
            synthWBimages[:, :, :, i] = changeWB(I, mf, 9) # apply it!
        return synthWBimages, wb_pf

def changeWB(input, m, f):  # apply the given mapping function m to input image
    sz = np.shape(input) # get size of input image
    I_reshaped = np.reshape(input,(int(input.size/3),3),
                            order="F") # reshape input to be n*3 (n: total number of pixels)
    if f == 3: # if selected kernel is 3, then do nothing
        kernel_out = kernel3(I_reshaped)
    elif f == 9: # if selected kernel is 9, use kernel9 function to compute it
        kernel_out = kernelP9(I_reshaped)
    elif f == 11: # if selected kernel is 11, use kernel11 function to compute it
        kernel_out = kernelP11(I_reshaped)
    out = np.dot(kernel_out, m) # apply m to the input image after raising it the selected higher degree
    out = out.reshape(sz[0], sz[1], sz[2],order="F") # reshape output image back to the original image shape
    out = outOfGamutClipping(out) # clip out-of-gamut pixels
    out = cv2.cvtColor(out.astype('float32'), cv2.COLOR_RGB2BGR)
    return out


def kernel3(I):  # identity   kernel
    # kernel(r, g, b) = [r, g, b];
    return I


def kernelP9(I):  # 9-poly kernel
    # kernel(r, g, b) = [r, g, b, r2, g2, b2, rg, rb, gb];
    return (np.transpose((I[:,0], I[:,1], I[:,2], I[:,0] * I[:,0],
                           I[:,1] * I[:,1], I[:,2] * I[:,2], I[:, 0] * I[:, 1],
                           I[:, 0] * I[:, 2], I[:, 1] * I[:, 2])))


def kernelP11(I):  # 11-poly kernel
    # kernel(R, G, B) = [R, G, B, RG, RB, GB, R2, G2, B2, RGB, 1];
    return np.transpose(I[:,0], I[:,1], I[:,2] , I[:, 0] * I[:, 1],
                                 I[:, 0] * I[:, 2], I[:, 1] * I[:, 2], I[:,0] * I[:,0],
                                 I[:,1] * I[:,1], I[:,2] * I[:,2], I[:, 0] * I[:, 1] * I[:, 2],
                                 np.ones((I.shape[0], 1)))


def outOfGamutClipping(I):  # out-of-gamut clipping
    sz = np.shape(I) # get size of input image I
    I = I.reshape(int(I.size / 3), 3) # reshape it
    I[I > 1] = 1 # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0 # any pixel is below 0, clip it to 0
    I = np.reshape(I, [sz[0], sz[1], sz[2]]) # return I back to its normal shape
    return I

def im2double(im): #from uint8 (0->255) to double (0->1)
    # info = np.iinfo(im.dtype) # Get data type of the input image
    # return im.astype(np.float64)
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)