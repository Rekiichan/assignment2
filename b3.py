import cv2 as cv
import numpy as np
import matplotlib as mpl 
from matplotlib import pyplot as plt

Init_matrix = cv.imread('2_2.bmp')
crop_matrix = Init_matrix[0:512,0:512,0]
rows, cols = crop_matrix.shape
freq_matrix = np.zeros(256).astype(int)

for r in range(0,rows):
    for c in range(0,cols):
        freq_matrix[crop_matrix[r][c]] += 1

max = freq_matrix[0]
for i in range(1,256):
    if freq_matrix[i] > max:
        max = freq_matrix[i]

fig,ax = plt.subplots(1,1)
x_axis = np.linspace(0,255,20)
ax.hist(freq_matrix, bins = x_axis)
ax.set_title("Histogram")
ax.set_ylabel('freq')
ax.set_xlabel('value')
plt.show()
# np.savetxt('result.txt',freq_matrix)


