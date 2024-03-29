import cv2
import numpy
from flat_stack import stack_flat_images

def stack_average(stack):
    average = [0,0,0]
    for img in stack:
        img_avg = numpy.average(numpy.average(img, axis=1), axis=0)
        average += img_avg
    num_of_imgs = len(stack)
    average = average / num_of_imgs
    return average

def stack_sigma(stack):
    sigma = numpy.std(numpy.std(numpy.std(stack, axis=2), axis=1), axis=0)
    return sigma

def reject_deviant_pixels(stack, mean, sigma, kappa):
    for img in range(len(stack)):
        print(img)
        for x in range(len(stack[img])):
            for y in range(len(stack[img][x])):
                pixel = stack[img][x, y]
                if abs(numpy.array([1, 0, 0]).dot(pixel - mean)) > (sigma[0] * kappa):
                    pixel = mean
                stack[img][x, y] = pixel


# Kappa sigma clipping for flat image stacking, source => http://deepskystacker.free.fr/english/technical.htm
# dark_images - array of flat images (numpy arrays)
def stack_dark_images(dark_images, dimensions, iterations, kappa):
    for i in range(iterations):
        mean = stack_average(dark_images)
        sigma = stack_sigma(dark_images)
        reject_deviant_pixels(dark_images, mean, sigma, kappa)
    return numpy.mean(dark_images, axis=0)







