import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('2_1.jpg')
mat = img / 255
rows, cols, high = mat.shape
V_matrix = np.zeros((rows,cols,high)).astype(float)

hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
v_img = np.zeros_like(v)

for r in range(v_img.shape[0]):
    for c in range(v_img.shape[1]):
        v_img[r,c] = 255*(v[r,c] - np.min(v))/(np.max(v)-np.min(v))

hsv_img = cv.merge([h,s,v_img])
bgr_img = cv.cvtColor(hsv_img,cv.COLOR_HSV2BGR)

cv.imshow('bgr_img',bgr_img)
cv.waitKey()
cv.destroyAllWindows()