import rawpy
import numpy as np
import os
import cv2 as cv

def subtract_dark(light_image, dark_image):
    outputImage = np.subtract(dark_image, light_image)
    return outputImage