from __future__ import division
import imageio
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg
import os

def compute_laplacian(img):
    eps = 1e-7
    img_h, img_w, img_d = img.shape
    idx_mat = np.arange(img_h * img_w).reshape((img_h, img_w))
    shape = (img_h - 3 + 1, img_w - 3 + 1, 3, 3)
    strides = (idx_mat.strides[0], idx_mat.strides[1]) + idx_mat.strides
    win_idx = as_strided(idx_mat, shape=shape, strides=strides).reshape(-1, 9)
    img = img.reshape(img_h * img_w, img_d)
    winI = img[win_idx]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.array([np.cov(win.T) for win in winI]) / 9 * 8
    inv = np.linalg.inv(win_var + (eps / 9) * np.eye(3)) # element-wise add

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv) # transpose and do dot product on last two axis
    X = np.einsum('...ij,...kj->...ik', X, winI - win_mu) # dot product
    L = np.eye(9) - (1 + X) / 9
    
    row_idx = np.repeat(win_idx, 9).ravel()
    col_idx = np.tile(win_idx, 9).ravel()
    return scipy.sparse.coo_matrix((L.flatten(), (row_idx, col_idx)))

def closed_form_matting(image, trimap):
    constraint_px = (trimap < 0.1) | (trimap > 0.9)
    Lambda = 100.0
    laplacian = compute_laplacian(image)
    alpha = scipy.sparse.linalg.spsolve(
        laplacian + scipy.sparse.diags((Lambda * constraint_px).flatten()),
        trimap.flatten() * (Lambda * constraint_px).flatten())
    alpha = np.minimum(np.maximum(alpha.reshape(trimap.shape), 0), 1)
    return alpha

IMG_PATH = "test/source.png"
TRIMAP_PATH = "test/trimap.png"
OUTPUT_PATH = "test/alpha.png"

image = imageio.imread(IMG_PATH) / 255.0
trimap = imageio.imread(TRIMAP_PATH) / 255.0
alpha = closed_form_matting(image, trimap)
imageio.imwrite(OUTPUT_PATH, alpha * 255.0)
