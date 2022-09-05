import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def process(src):
    converted_image = np.zeros((src.shape))
    converted_image = 255 - src
    return converted_image


if __name__ == "__main__":
    src = cv.imread("2_2.bmp")
    converted_image = process(src)
    cv.imshow("original", src)
    cv.imshow("converted", converted_image)
    cv.waitKey()
    cv.destroyAllWindows()