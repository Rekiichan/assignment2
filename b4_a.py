import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def linear_stretching(src):
    rgb_src_img = cv.cvtColor(src,cv.COLOR_BGR2RGB)
    hsv_img = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    h,s,v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    V_matrix = np.zeros_like(v)

    for r in range(V_matrix.shape[0]):
        for c in range(V_matrix.shape[1]):
            V_matrix[r,c] = 255*(v[r,c] - np.min(v))/(np.max(v)-np.min(v))

    hsv_img = cv.merge([h,s,V_matrix])
    bgr_img = cv.cvtColor(hsv_img,cv.COLOR_HSV2BGR)
    rgb_cvt_img = cv.cvtColor(bgr_img,cv.COLOR_BGR2RGB)
    return rgb_src_img, rgb_cvt_img

def get_histogram(src):
    rgb_src_img = cv.cvtColor(src,cv.COLOR_BGR2RGB)
    hsv_img = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    gray_img = hsv_img[:,:,2]
    rows,cols = src.shape[0],gray_img.shape[1]
    freq = np.zeros(256).astype(int)

    for r in range(gray_img.shape[0]):
        for c in range(gray_img.shape[1]):
            freq[gray_img[r][c]] += 1

    return freq

def get_cdf(src):
    rgb_src_img = cv.cvtColor(src,cv.COLOR_BGR2RGB)
    hsv_img = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    gray_img = hsv_img[:,:,2]

    freq_matrix = get_histogram(src)
    rows,cols = gray_img.shape[0],gray_img.shape[1]
    p = freq_matrix / (rows*cols)
    cdf = np.zeros((256))

    for i in range (len(cdf)):
        cdf[i] = np.sum(p[:i])     

    return cdf

def display_img(original,converted_img):
    axis_X = np.arange(0, 256, 1)
    hist_ori = get_histogram(original)
    cdf_ori = get_cdf(original)
    hist_cvt = get_histogram(converted_img)
    cdf_cvt = get_cdf(converted_img)
    fig = plt.figure(figsize=(11, 7))

    fig.add_subplot(2, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis("off")

    fig.add_subplot(2, 3, 2)
    plt.bar(axis_X,hist_ori)
    plt.title('Histogram')

    fig.add_subplot(2, 3, 3)
    plt.bar(axis_X,cdf_ori)
    plt.title('CDF')

    fig.add_subplot(2, 3, 4)
    plt.imshow(converted_img)
    plt.title('Converted Image')
    plt.axis("off")

    fig.add_subplot(2, 3, 5)
    plt.bar(axis_X,hist_cvt)
    plt.title('Histogram')

    fig.add_subplot(2, 3, 6)
    plt.bar(axis_X,cdf_cvt)
    plt.title('CDF')

    plt.show()

if __name__ == '__main__':
    img_2_1_src = cv.imread('2_1.jpg')
    img_2_1_src, img_2_1_cvt = linear_stretching(img_2_1_src)
    img_2_3_src = cv.imread('2_3.jpg')
    img_2_3_src, img_2_3_cvt = linear_stretching(img_2_3_src)
    display_img(img_2_1_src, img_2_1_cvt)
    display_img(img_2_3_src, img_2_3_cvt)
