# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize

IMG_ROOT = "rgb_img"
ALPHA_KNN_ROOT = "alpha_knn"
ALPHA_CF_ROOT = "alpha_cf"
ALPHA_GT_ROOT = "alpha_gt"
OUTPUT_ROOT = "npz_s"

PATCH_SIZE = 32 # same as cifar-10

index = [13, 17, 20]

for idx in index:
    print("idx: %d" % (idx))

    IMG_PATH =       os.path.join(IMG_ROOT,       str(idx))
    ALPHA_KNN_PATH = os.path.join(ALPHA_KNN_ROOT, str(idx))
    ALPHA_CF_PATH =  os.path.join(ALPHA_CF_ROOT,  str(idx))

    # ground truth alpha
    alpha_gt = cv2.imread(os.path.join(ALPHA_GT_ROOT, "GT%02d.png" % (idx)), cv2.IMREAD_GRAYSCALE)
    alpha_gt = alpha_gt.reshape(alpha_gt.shape[0], alpha_gt.shape[1], 1)
    alpha_gt = alpha_gt / 255.0

    alpha_cf_files =  [f for f in os.listdir(ALPHA_CF_PATH) if os.path.isfile(os.path.join(ALPHA_CF_PATH, f))]
    alpha_knn_files = [f for f in os.listdir(ALPHA_KNN_PATH) if os.path.isfile(os.path.join(ALPHA_KNN_PATH, f))]

    train_data = []

    for filename in alpha_cf_files:

        if filename in alpha_knn_files:
            # closed-form alpha
            alpha_cf = cv2.imread(os.path.join(ALPHA_CF_PATH, filename), cv2.IMREAD_GRAYSCALE) / 255.0
            alpha_cf = alpha_cf.reshape(alpha_cf.shape[0], alpha_cf.shape[1], 1)
            alpha_cf = alpha_cf / 255.0

            # knn alpha
            alpha_knn = cv2.imread(os.path.join(ALPHA_KNN_PATH, filename), cv2.IMREAD_GRAYSCALE) / 255.0
            alpha_knn = alpha_knn.reshape(alpha_knn.shape[0], alpha_knn.shape[1], 1)
            alpha_knn = alpha_knn / 255.0


            # rgb image
            rgb_img = cv2.imread(os.path.join(IMG_PATH, filename))
            # normalize image
            rgb_img = rgb_img.reshape((-1, 3))
            normalize(rgb_img, norm="l2", axis=1)
            rgb_img = rgb_img.reshape((alpha_cf.shape[0], alpha_cf.shape[1], 3))


            # concatenate three inputs and ground truth alpha
            concat_img = np.concatenate((rgb_img, alpha_knn, alpha_cf, alpha_gt), axis=-1)

            row_strides = [i*PATCH_SIZE for i in range(int(concat_img.shape[0]/PATCH_SIZE))]
            col_strides = [i*PATCH_SIZE for i in range(int(concat_img.shape[1]/PATCH_SIZE))]
            for row_idx in row_strides:
                for col_idx in col_strides:
                    patch = concat_img[row_idx:row_idx+PATCH_SIZE, col_idx:col_idx+PATCH_SIZE]
                    if not (np.mean(patch[:,:,-1]) == 0 or np.mean(patch[:,:,-1]) == 1):
                        train_data.append(patch)
                
    print("data length: %d" % (len(train_data)))
    save_file = os.path.join(OUTPUT_ROOT, "%d.npz" %(idx))
    np.save(save_file, train_data)
    train_data.clear()
    del train_data[:]
