import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

init_matrix = cv.imread('2_2.bmp')
cropped_matrix = init_matrix[0:512,0:512,0]
rows, cols = cropped_matrix.shape
freq_matrix = np.zeros(256).astype(int)

for r in range(0,rows):
    for c in range(0,cols):
        freq_matrix[cropped_matrix[r][c]] += 1
sum = 0
for r in range(0,rows):
    for c in range(0,cols):
        sum += 1

# print(sum)

axis_X = np.arange(0, 256, 1)

plt.plot(freq_matrix)
plt.bar(axis_X,freq_matrix)
plt.ylabel("value")
plt.show()

p = freq_matrix / (rows*cols)
cdf = np.zeros((256))
for i in range (len(cdf)):
    cdf[i] = np.sum(p[:i])              

plt.plot(cdf)
plt.title('The cumulative distribution function (cdf)')
plt.bar(axis_X,cdf)
plt.xlabel('Pixel Value')
plt.ylabel('Probability')
plt.title('Nomalized Histogram')
plt.show()
# np.savetxt('result.txt',freq_matrix)
