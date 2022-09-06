import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

Init_matrix = cv.imread('2_2.bmp')
crop_matrix = Init_matrix[0:512,0:512,0]
rows, cols = crop_matrix.shape
freq_matrix = np.zeros(256).astype(int)

for r in range(0,rows):
    for c in range(0,cols):
        freq_matrix[crop_matrix[r][c]] += 1

axis_X = np.arange(0, 256, 1)
plt.bar(axis_X,freq_matrix)
plt.ylabel("value")
plt.show()
# np.savetxt('result.txt',freq_matrix)
