from __future__ import division
import imageio
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg
import os

def _rolling_block(A, block=(3, 3)):
    # Applies sliding window to given matrix.
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def compute_laplacian(img, mask=None, eps=10**(-7), win_rad=1):
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1
    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))
    win_inds = win_inds.reshape(c_h, c_w, win_size)
    win_inds = win_inds.reshape(-1, win_size)
    winI = ravelImg[win_inds] # (43571, 9, 3)

    win_mu = np.mean(winI, axis=1, keepdims=True)
    # print win_mu.shape # (43571, 3)
    # print win_var.shape # (43571, 3, 3)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)
    # win_var = np.array([np.cov(win.T) for win in winI]) / 9 * 8

    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3)) # element-wise add
    # print inv.shape # (43571, 3, 3)

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv) # transpose and do dot product on last two axis
    vals = np.eye(win_size) - (1/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))
    # print vals.shape # (43571, 9, 9)


    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)))
    # print L.shape # (44415, 44415)
    return L

def closed_form_matting(image, trimap):
    constraint_px = (trimap < 0.1) | (trimap > 0.9)
    Lambda = 100.0
    laplacian = compute_laplacian(image)
    alpha = scipy.sparse.linalg.spsolve(
        laplacian + scipy.sparse.diags((Lambda * constraint_px).ravel()),
        trimap.ravel() * (Lambda * constraint_px).ravel())
    alpha = np.minimum(np.maximum(alpha.reshape(trimap.shape), 0), 1)
    return alpha


todos = [117, 136, 155, 174]

for todo in todos:
    IMG_PATH = "/media/lobst3rd/DATA/dataset/matting/output/1/%d.png" % (todo)
    TRIMAP_PATH = "/media/lobst3rd/DATA/dataset/matting/train/trimap_training_lowres/Trimap1/GT01.png"
    OUTPUT_PATH = "/media/lobst3rd/DATA/dataset/matting/numpy/alpha_cf/1/%d.png" % (todo)

    image = imageio.imread(IMG_PATH) / 255.0
    trimap = imageio.imread(TRIMAP_PATH) / 255.0
    alpha = closed_form_matting(image, trimap)
    imageio.imwrite(OUTPUT_PATH, alpha * 255.0)
