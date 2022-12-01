import numpy as np

def subtract_bias(input_image, flat_image):
    outputImage = np.subtract(input_image, flat_image)
    return outputImage