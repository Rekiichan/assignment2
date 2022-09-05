import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

Init_matrix = cv.imread('2_2.bmp')
float_matrix = np.array(Init_matrix).astype(float)

LSB = float_matrix % 2
floor_1st = np.floor(float_matrix / 2) % 2
floor_2nd = np.floor(float_matrix / 4) % 2
floor_3rd = np.floor(float_matrix / 8) % 2
floor_4th = np.floor(float_matrix / 16) % 2
floor_5th = np.floor(float_matrix / 32) % 2
floor_6th = np.floor(float_matrix / 64) % 2
MSB = np.floor(float_matrix / 128) % 2

Rec_Image = (2 * (2 * (2 * (2 * (2 * (2 * (2 * MSB + floor_6th) + floor_5th)
+ floor_4th) + floor_3rd) + floor_3rd) + floor_2nd) + LSB)

fig = plt.figure(figsize=(12, 9))

fig.add_subplot(2, 5, 1)
plt.imshow(Init_matrix)
plt.title('Original Image')
plt.axis("off")
  
fig.add_subplot(2, 5, 2)
plt.imshow(LSB)
plt.title('Bit Plane 1')
plt.axis("off")

fig.add_subplot(2, 5, 3)
plt.imshow(floor_1st)
plt.title('Bit Plane 2')
plt.axis("off")

fig.add_subplot(2, 5, 4)
plt.imshow(floor_2nd)
plt.title('Bit Plane 3')
plt.axis("off")

fig.add_subplot(2, 5, 5)
plt.imshow(floor_3rd)
plt.title('Bit Plane 4')
plt.axis("off")

fig.add_subplot(2, 5, 6)
plt.imshow(floor_4th)
plt.title('Bit Plane 5')
plt.axis("off")

fig.add_subplot(2, 5, 7)
plt.imshow(floor_5th)
plt.title('Bit Plane 6')
plt.axis("off")

fig.add_subplot(2, 5, 8)
plt.imshow(floor_6th)
plt.title('Bit Plane 7')
plt.axis("off")

fig.add_subplot(2, 5, 9)
plt.imshow(MSB)
plt.title('Bit Plane 8')
plt.axis("off")

fig.add_subplot(2, 5, 10)
Rec_Image = Rec_Image.astype(int)
plt.imshow(Rec_Image)
plt.title('Recombined Image')
plt.axis("off")

plt.show()
