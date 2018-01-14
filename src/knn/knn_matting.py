#!/usr/bin/env python
# -*- coding: utf-8 -*-
#in the numpy system, the shape is [row,column]

import cv2 
import numpy as np
from sklearn.neighbors import NearestNeighbors  
import scipy.sparse
import warnings
import os 

def knn_matte(IMG, IMG2, mylambda=1000):
    #get the shape value of the image 
    [r,c,l] = IMG.shape

    # translate the rgb image to the HSV
    print("change the rgb to the hsv ")
    #R,G,B = cv2.split(IMG)
    HSV = cv2.cvtColor(IMG,cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(HSV)
    H,S,V = H / 180.0 ,S / 255.0 ,V / 255.0 
    k = np.dstack((H,S,V)) #merge the HSV

    #set the constrain 
    IMG2 = IMG2 / 255.0 
    foreground = (IMG2 > 0.99).astype(int)
    background = (IMG2 < 0.01).astype(int)
    constrain = foreground + background

    #set the flatten item index (x,y)
    x, y = np.unravel_index(np.arange(r*c), (r, c)) # save the array index 
    H_cos = np.cos(H)
    H_sin = np.sin(H)
    H = np.append(H_cos.reshape(r*c,1).T,H_sin.reshape(r*c,1).T,axis = 0)
    SV = np.append(S.reshape(r*c,1).T,V.reshape(r*c,1).T,axis = 0)
    P = [x, y]/np.sqrt(r*c + c*c) #position (x, y)
    feature_vec = np.append(H,SV,axis = 0)
    feature_vec = np.append(feature_vec,P,axis = 0).T
    nbrs = NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    # Compute Sparse A
    print('Computing sparse A')
    row_inds = np.repeat(np.arange(r*c), 10)
    col_inds = knns.reshape(r*c*10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1)/(6) 
    #6 is the paper C

    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)),shape=(r*c, r*c))
    d = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = d-A

    print(L.shape)
    D = scipy.sparse.diags(np.ravel(constrain[:,:, 0]))
    v = np.ravel(foreground[:,:,0])
    C = 2*mylambda*np.transpose(v)
    h = 2*(L + mylambda*D)

    print('Solving linear system for alpha')
    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(h, C), 0), 1).reshape(r, c)
    except Warning:
        x = scipy.sparse.linalg.lsqr(h, C)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(r, c)
    return alpha



print("read image file and the tri map ")
#read the origin img and the trimap 
# arr_1 = os.listdir("output")
# for item_file in arr_1:
#     print(item_file)
#     input()
#     count = 0
#     if(item_file < 10 ):
#         IMG2 = cv2.imread("Trimap1/GT0"+item_file+".png")
#     else:
#         IMG2 = cv2.imread("Trimap1/GT"+item_file+".png")
#     arr_2 = os.listdir('output/'+item_file)
#     for item in arr_2 :
#         print(IMG2.shape)
#         print(item)
#         if count < 100 :
#             IMG = cv2.imread('output/'+item_file+"/"+item)
#             # IMG = cv2.resize(IMG,(IMG2.shape[1],IMG2.shape[0]))
#             print(IMG.shape)
#             alpha = knn_matte(IMG, IMG2)
#             cv2.imwrite("knn_output/" + item_file + "/" + item,alpha * 255)
#         count+=1
CF_DIR = "numpy/alpha_cf"
DATA_DIR = "output"
Trimap_DIR = "train/trimap_training_lowres/Trimap1"
for folder in os.listdir(CF_DIR):
    print(folder)
    sub_CF_DIR = CF_DIR+'/'+folder
    sub_DATA_DIR = DATA_DIR+'/'+folder
    if int(folder)==25:
        full_trimap_path = Trimap_DIR + "/GT"+folder+".png"
        trimap = cv2.imread(full_trimap_path)
    else:
        continue
        # full_trimap_path = "Trimap1/GT" + folder + ".png"
        # trimap = cv2.imread(full_trimap_path)
    for file in os.listdir(sub_CF_DIR):
        
        if not file in os.listdir("knn_output/" + folder):
            print(file)
            full_src_path = sub_DATA_DIR+'/'+file
            img = cv2.imread(full_src_path,1)
            #cv2.imshow("img",img)
            #cv2.waitKey(0)
            alpha = knn_matte(img, trimap)
            cv2.imwrite("knn_output/" + folder + "/" + file, alpha * 255)
