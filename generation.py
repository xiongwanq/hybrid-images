# -*- coding:utf-8 -*-
# @Time       :2022/11/27 17:05
# @AUTHOR     :XiongWanqing
# @SOFTWARE   :hybrid-images
# @task       :hybrid images
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 使图像大小相同
def InitImgs(img1, img2):
    rows, cols = img1.shape[:2]
    img2 = cv.resize(img2,(cols,rows),interpolation=cv.INTER_CUBIC)
    return img1, img2
# 并排显示原图像+频谱图
def show(img, fimg):
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original image'), plt.axis('off')
    plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier image'), plt.axis('off')
    plt.show()
# 获得频谱图
def GetFourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift))
    return fimg
# 像素点到图像中心的距离
def Distance(m,n):
    u = np.array([i - m / 2 for i in range(m)], dtype=np.float32)
    v = np.array([i - n / 2 for i in range(n)], dtype=np.float32)
    ret = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            ret[i][j] = np.sqrt(u[i] * u[i] + v[j] * v[j])
    return ret
# 巴特沃斯低通滤波
def GetMatrix(img,fimg,n,d0):
    duv = Distance(*fimg.shape[:2])
    filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
    return filter_mat

img1 = cv.imread('selfie.jpg', 0)
img2 = cv.imread('network_photo.jpg', 0)
img1, img2 = InitImgs(img1,img2)
# 频谱图1
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
fimg1 = np.log(np.abs(fshift1))
show(img1,fimg1)
# 频谱图2
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
fimg2 = np.log(np.abs(fshift2))
show(img2,fimg2)

# 自拍的低频
low = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift1 * GetMatrix(img1,fimg1,2,20))))
show(low, GetFourier(low))
# 明星的高频
high = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift2 * (1-GetMatrix(img2,fimg2,2,10)))))# 10 20
show(high, GetFourier(high))
# 两者融合
hybrid = low + high
show(hybrid, GetFourier(hybrid))