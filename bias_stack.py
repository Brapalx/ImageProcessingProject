import cv2
import numpy

# mean stacking of bias images
# bias_images - stack of bias images
def stack_bias_images(bias_images, dimensions):
    avg_image = bias_images[0]

    for i in range(len(bias_images)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i+1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(bias_images[i], alpha, avg_image, beta, 0.0)

    return avg_image
