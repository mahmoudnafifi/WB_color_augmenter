import numpy as np
import numpy.matlib
import cv2
import os

def trythis(input_image, output_dir):
    filename, file_extension = os.path.splitext(input_image)
    features = np.load('WBEmulator/features.npy')  # training encoded features
    mappingFuncs = np.load('WBEmulator/mappingFuncs.npy')  # mapping functions to emulate WB effects
    encoderWeights = np.load('WBEmulator/encoderWeights.npy')  # weight matrix for histogram encoding
    encoderBias = np.load('WBEmulator/encoderBias.npy')  # bias vector for histogram encoding
    h = 60  # histogram bin width
    K = 25  # K value for nearest neighbor searching
    sigma = 0.25  # fall off factor for KNN
    wb_photo_finishing = ['_F_AS', '_F_CS', '_S_AS', '_S_CS',
                               '_T_AS', '_T_CS', '_C_AS', '_C_CS',
                               '_D_AS', '_D_CS']  # postfix for the generated images

    I = cv2.imread(input_image) # read the image
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
    I = im2double(I) # convert to double
    I_orig = I # take a backup image


    # compute RGB-uv histogram
    sz = np.shape(I) # get size of current image
    if sz[0] * sz[1] > 202500:  # if it is larger than 450*450
        factor = np.sqrt(202500 / (sz[0] * sz[1])) # rescale factor
        newH = int(np.floor(sz[0] * factor))
        newW = int(np.floor(sz[1] * factor))
        I= cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST) # resize image
    II = I.reshape(int(I.size/3), 3)  # n*3
    inds = np.where((II[:,0] >0) & (II[:,1]>0) & (II[:,2]>0)) # remove any zero pixels
    R = II[inds,0] # red channel
    G = II[inds,1] # green channel
    B = II[inds,2] # blue channel
    I_reshaped = np.concatenate((R, G, B),axis=0).transpose() # reshaped image (wo zero values)
    eps = 6.4 / h
    A = np.arange(-3.2, 3.19, eps)  # dummy vector
    hist = np.zeros((A.size, A.size, 3)) # histogram will be stored here
    Iy = np.sqrt(np.power(I_reshaped[:, 0], 2) + np.power(I_reshaped[:, 1], 2) +
                 np.power(I_reshaped[:, 2], 2)) # intensity vector
    for i in range(3): # for each histogram layer, do
        r = [] # excluded channels will be stored here
        for j in range(3): # for each color channel do
            if j != i: # if current color channel does not match current histogram layer,
                r.append(j) # exclude it
        Iu = np.log(I_reshaped[:, i]/I_reshaped[:, r[1]]) # current color channel / the first excluded channel
        Iv = np.log(I_reshaped[:, i]/I_reshaped[:, r[0]]) # current color channel / the second excluded channel
        diff_u = np.abs(np.matlib.repmat(Iu, np.size(A), 1).transpose() - np.matlib.repmat(A, np.size(Iu), 1)) # differences in u space
        diff_v = np.abs(np.matlib.repmat(Iv, np.size(A), 1).transpose() - np.matlib.repmat(A, np.size(Iv), 1)) # differences in v space
        diff_u[diff_u >= (eps / 2)] = 0 # do not count any pixel has difference beyond the threshold in the u space
        diff_u[diff_u != 0] = 1 # remaining pixels will be counted
        diff_v[diff_v >= (eps / 2)] = 0 # do not count any pixel has difference beyond the threshold in the v space
        diff_v[diff_v != 0] = 1 # remaining pixels will be counted
        temp = (np.matlib.repmat(Iy, np.size(A), 1) * (diff_u).transpose()) # Iy * diff_u'
        hist[:, :, i] = np.dot(temp,diff_v) # initialize current histogram layer with Iy * diff' * diff_v
        norm_ = np.sum(hist[:, :, i],axis=None) # compute norm value
        hist[:,:,i] = np.sqrt(hist[:,:,i]/norm_) # (hist/norm)^(1/2)

    histR_reshaped = np.reshape(np.transpose(hist[:,:,0]),
                                (1, int(hist.size/3)), order="F") # reshaped red layer of histogram
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                (1, int(hist.size / 3)), order="F") # reshaped green layer of histogram
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                (1, int(hist.size / 3)), order="F") # reshaped blue layer of histogram
    hist_reshaped = np.append(histR_reshaped,[histG_reshaped, histB_reshaped]) # reshaped histogram n * 3 (n = h*h)
    feature = np.dot(hist_reshaped - encoderBias.transpose(), encoderWeights) # compute compacted histogram feature

    outNum = len(wb_photo_finishing) # number of images to generate
    I = cv2.normalize(I.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) # do not know why!
    if outNum < len(wb_photo_finishing): # if selected outNum is less than number of available WB & photo finishing (PF) styles,
        inds = np.random.permutation(len(wb_photo_finishing)) # randomize and select outNum WB & PF styles
        wb_pf = wb_photo_finishing[inds[0:outNum - 1]] # wb_pf now represents the selected WB & PF styles
    else: # if the selected number of images equals the available WB & PF styles,
        wb_pf = wb_photo_finishing # then wb_pf = wb_photo_finishing
        inds = list(range(0, len(wb_pf))) # inds is simply from 0 to the number of available WB & PF styles
    synthWBimages = np.zeros((I_orig.shape[0], I_orig.shape[1],
                              I_orig.shape[2], len(wb_pf))) # synthetic images will be stored here


    D_sq = np.einsum('ij, ij ->i', features, features)[:, None] + \
           np.einsum('ij, ij ->i', feature, feature) - 2 * features.dot(feature.T) # squared euclidean distances

    idH = D_sq.argpartition(K, axis=0)[:K] # get smallest K distances
    dH = np.sqrt(np.take_along_axis(D_sq, idH, axis=0)) # square root nearest distances to get real euclidean distances
    sorted_idx = dH.argsort(axis=0) # get sorting indices
    idH = np.take_along_axis(idH, sorted_idx, axis=0) # sort distance indices
    dH = np.take_along_axis(dH, sorted_idx, axis=0) # sort distances

    weightsH = np.exp(-(np.power(dH,2))/(2*np.power(sigma,2))) # compute blending weights
    weightsH = weightsH / sum(weightsH) # normalize blending weights
    for i in range(len(inds)): # for each of the retried training examples, do
        ind = inds[i] # for each WB & PF style,
        mf = sum(np.reshape(np.matlib.repmat(weightsH,1,27),(25,1,9,3)) *
                 mappingFuncs[(idH - 1) * 10 + ind, :]) # compute the mapping function
        mf = mf.reshape(9, 3,order="F") # reshape it to be 9 * 3
        synthWBimages[:, :, :, i] = changeWB(I_orig, mf, 9) # apply it!
        outImg = synthWBimages[:,:,:,i]
        outImg = cv2.cvtColor(outImg.astype('float32'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_dir + '/' + os.path.basename(filename) + '_' + wb_pf[i] + file_extension, outImg * 255)

    return synthWBimages


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
    return out


def kernel3(I):  # identity kernel
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
    info = np.iinfo(im.dtype) # Get data type of the input image
    return im.astype(np.float64) / info.max # Divide all values by the largest value in the datatype