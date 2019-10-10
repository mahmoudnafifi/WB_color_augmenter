import numpy as np
import numpy.matlib
import cv2
import os

def backup_code(input_image, output_dir):
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

    I = cv2.imread(input_image)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = im2double(I)
    I_orig = I

    sz = np.shape(I)
    if sz[0] * sz[1] > 202500:  # 450*450
        factor = np.sqrt(202500 / (sz[0] * sz[1]))
        newH = int(np.floor(sz[0] * factor))
        newW = int(np.floor(sz[1] * factor))
        print(newH,newW)
        I= cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)
    II = I.reshape(int(I.size/3), 3)  # n*3
    inds = np.where((II[:,0] >0) & (II[:,1]>0) & (II[:,2]>0))
    R = II[inds,0]
    G = II[inds,1]
    B = II[inds,2]
    a = np.concatenate((R, G, B),axis=0).transpose()
    eps = 6.4 / 60
    A = np.arange(-3.2, 3.19, eps)  # dummy vector
    hist = np.zeros((A.size, A.size, 3))
    Iy = np.sqrt(np.power(a[:, 0], 2) + np.power(a[:, 1], 2) + np.power(a[:, 2], 2))
    for i in range(3):
        r = []
        for j in range(3):
            if j != i:
                r.append(j)
        Iu = np.log(a[:, i]/a[:, r[1]])
        Iv = np.log(a[:, i]/a[:, r[0]])
        diff_u = np.abs(np.matlib.repmat(Iu, np.size(A), 1).transpose() - np.matlib.repmat(A, np.size(Iu), 1))
        diff_v = np.abs(np.matlib.repmat(Iv, np.size(A), 1).transpose() - np.matlib.repmat(A, np.size(Iv), 1))
        diff_u[diff_u >= (eps / 2)] = 0
        diff_u[diff_u != 0] = 1
        diff_v[diff_v >= (eps / 2)] = 0
        diff_v[diff_v != 0] = 1
        temp = (np.matlib.repmat(Iy, np.size(A), 1) * (diff_u).transpose())
        hist[:, :, i] = np.dot(temp,diff_v)
        t = hist[:,:,i]
        norm_ = np.sum(t,axis=None)
        hist[:,:,i] = np.sqrt(hist[:,:,i]/norm_)



    histR_reshaped = np.reshape(np.transpose(hist[:,:,0]),(1, int(hist.size/3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]), (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]), (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped,[histG_reshaped, histB_reshaped])
    feature = np.dot(hist_reshaped - encoderBias.transpose(), encoderWeights)

    outNum = len(wb_photo_finishing)
    I = cv2.normalize(I.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    if outNum < len(wb_photo_finishing):
        inds = np.random.permutation(len(wb_photo_finishing))
        wb_pf = wb_photo_finishing[inds[0:outNum - 1]]
    else:
        wb_pf = wb_photo_finishing
        inds = list(range(0, len(wb_pf)))
    synthWBimages = np.zeros((I_orig.shape[0], I_orig.shape[1], I_orig.shape[2], len(wb_pf)))


    D_sq = np.einsum('ij, ij ->i', features, features)[:, None] + \
           np.einsum('ij, ij ->i', feature, feature) - 2 * features.dot(feature.T)

    idH = D_sq.argpartition(K, axis=0)[:K]
    dH = np.sqrt(np.take_along_axis(D_sq, idH, axis=0))
    sorted_idx = dH.argsort(axis=0)
    idH = np.take_along_axis(idH, sorted_idx, axis=0)
    dH = np.take_along_axis(dH, sorted_idx, axis=0)

    weightsH = np.exp(-(np.power(dH,2))/(2*np.power(sigma,2)))
    weightsH = weightsH / sum(weightsH)
    for i in range(len(inds)):
        ind = inds[i]
        mf = sum(np.reshape(np.matlib.repmat(weightsH,1,27),(25,1,9,3)) * mappingFuncs[(idH - 1) * 10 + ind, :])
        mf = mf.reshape(9, 3,order="F")
        synthWBimages[:, :, :, i] = changeWB(I_orig, mf, 9)
        outImg = synthWBimages[:,:,:,i]
        outImg = cv2.cvtColor(outImg.astype('float32'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_dir + '/' + os.path.basename(filename) + '_' + wb_pf[i] + file_extension, outImg * 255)

    return synthWBimages


def changeWB(input, m, f):  # apply given m to input
    sz = np.shape(input)

    I_reshaped = np.reshape(input,(int(input.size/3),3),order="F")
    if f == 3:
        kernel_out = kernel3(I_reshaped)
    elif f == 9:
        kernel_out = kernelP9(I_reshaped)
    elif f == 11:
        kernel_out = kernelP11(I_reshaped)
    out = np.dot(kernel_out, m)
    out = out.reshape(sz[0], sz[1], sz[2],order="F")
    out = outOfGamutClipping(out)
    return out


def kernel3(I):  # idle kernel
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
    sz = np.shape(I)
    I = I.reshape(int(I.size / 3), 3)
    I[I > 1] = 1
    I[I < 0] = 0
    I = np.reshape(I, [sz[0], sz[1], sz[2]])
    return I

    #print(np.shape(a))
    #cv2.imshow('image', hist * 10)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite(output_image, hist * 10 * 255)
    #sp = np.shape(I)
    #print(sp[0] * sp[1])



def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float64) / info.max # Divide all values by the largest possible value in the datatype