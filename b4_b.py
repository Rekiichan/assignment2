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
    return freq,gray_img

def hist_equal(src):
    freq,gray_img = get_histogram(src)
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
    hsv_img = cv.cvtColor(src,cv.COLOR_BGR2HSV)
    h,s = hsv_img[:,:,0], hsv_img[:,:,1]
    hsv_img = cv.merge([h,s,gray_img])
    bgr_img = cv.cvtColor(hsv_img,cv.COLOR_HSV2BGR)
    rgb_cvt_img = cv.cvtColor(bgr_img,cv.COLOR_BGR2RGB)
    return rgb_cvt_img

if __name__ == '__main__':
    img_2_1 = cv.imread('2_1.jpg')
    rgb_cvt_img = hist_equal(img_2_1)
