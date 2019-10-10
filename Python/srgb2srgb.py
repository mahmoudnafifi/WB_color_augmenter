import model_WB

def srgb2srgb(I):
    load('model.mat')
    out = model.correctImage(I)
    return out


