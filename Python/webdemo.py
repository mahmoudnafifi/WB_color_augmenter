import numpy as np
import numpy.matlib
import cv2

def webdemo(input_image, output_image):
    I = cv2.imread(input_image, 1)
    I = cv2.normalize(I.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    #print(img)
    sz = np.shape(I)
    if sz[0] * sz[1] > 202500:  # 450*450
        factor = np.sqrt(202500 / (sz[0] * sz[1]))
        newH = int(np.floor(sz[0] * factor))
        newW = int(np.floor(sz[1] * factor))
        print(newH,newW)
        I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)
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


    print(np.shape(a))
    #cv2.imshow('image', hist * 10)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_image, hist * 10 * 255)
    #sp = np.shape(I)
    #print(sp[0] * sp[1])
