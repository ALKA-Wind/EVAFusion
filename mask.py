import torch
import PIL.Image as Image
import numpy as np
# import matplotlib.pyplot as plt
import os
import cv2


def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

def calhis(img):
    ret = np.zeros(256, int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ret[img[i][j]] += 1
    return ret
def sal(his):
    ret = np.zeros(256, int)
    for i in range(256):
        for j in range(256):
            ret[i] += np.abs(j-i)*his[j]
    return ret

def vsm(img):
    his = np.zeros(256, int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i][j]] += 1
    sal = np.zeros(256, int)
    for i in range(256):
        for j in range(256):
            sal[i] += np.abs(j-i)*his[j]
    map = np.zeros_like(img, int)
    for i in range(256):
        map[np.where(img == i)] = sal[i]
    if map.max()==0:
        return np.zeros_like(img, int)
    return map / (map.max())

def softmax(map1, map2, c):
    exp_x1 = np.exp(map1*c)
    exp_x2 = np.exp(map2*c)
    exp_sum = exp_x1 + exp_x2
    map1 = exp_x1/exp_sum
    map2 = exp_x2/exp_sum
    # print(map1)
    # print(map2)
    return map1, map2


def map_generate(map1,map2,c=10):
    map1_ = vsm(map1)
    map2_ = vsm(map2)
    return softmax(map1_,map2_,c)

###############
## VSM for segmentation
##############
def map_generate2(map1,map2,c =20):
    # map1_ = vsm(map1)
    # map2_ = vsm(map2)
    # w1 = 0.4+ (map1_*1-map2_*0.4)

    map1_ = vsm(map1)  # 红外图像显著性
    map2_ = vsm(map2)  # 可见光图像显著性

    # 新的权重计算公式
    # 方法1: 使用最大值确保总是有一定的亮度
    # w1 = 0.4 + 0.5 * (map1_ - map2_*0.4)
    # w1 = 0.4 + 0.3 * (map1_ - map2_*0.6)
    w1 = 0.3 + 0.2 * (map1_ - map2_*0.8)

    # 确保权重在[0,1]范围内 - 这步非常重要
    # w1 = np.clip(w1, 0.05, 0.85)  # 确保权重永远不会接近0或1
    w1 = np.clip(w1, 0.05, 0.7)
    w2 = 1-w1
    return w1,w2
