import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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

def hist_equal(src):
    hsv_img = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    h,s,gray_img = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    freq = get_histogram(src)
    rows,cols = gray_img.shape
    size = rows*cols
    p = freq / size
    hist = np.zeros((256))
    for i in range(len(hist)):
        hist[i] = sum(freq[:i])
    max_val = max(hist)
    min_val = min(hist)
    hist = [int(((f-min_val)/(max_val-min_val))*255) for f in hist]
    for row in range(gray_img.shape[0]): # traverse by row (y-axis)
        for col in range(gray_img.shape[1]): # traverse by column (x-axis)
            gray_img[row, col] = hist[gray_img[row, col]]

    hsv_img = cv.merge([h,s,gray_img])
    bgr_img = cv.cvtColor(hsv_img,cv.COLOR_HSV2BGR)
    rgb_cvt_img = cv.cvtColor(bgr_img,cv.COLOR_BGR2RGB)
    return rgb_cvt_img

def display_img(original,converted_img):
    hist_ori = get_histogram(original)
    hist_cvt = get_histogram(converted_img)
    cdf_ori = get_cdf(original)
    cdf_cvt = get_cdf(converted_img)
    axis_X = np.arange(0, 256, 1)
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
    img_2_1 = cv.imread('2_1.jpg')
    rgb_src_img1 = cv.cvtColor(img_2_1,cv.COLOR_BGR2RGB)
    rgb_cvt_img1 = hist_equal(img_2_1)
    display_img(rgb_src_img1,rgb_cvt_img1)
    img_2_3 = cv.imread('2_3.jpg')
    rgb_src_img2 = cv.cvtColor(img_2_3,cv.COLOR_BGR2RGB)
    rgb_cvt_img2 = hist_equal(img_2_3)
    display_img(rgb_src_img2,rgb_cvt_img2)