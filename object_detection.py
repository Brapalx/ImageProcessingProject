import numpy


# Creates a mask of an image by looking at pixel "brightness" to notice stars
# Work in progress, output needs to be cleaned
# image - numpy array for image to generate mask from
# dimensions - array of [ width , height , # of color channels ]
# return = an image with 255 for part of an object, 0 for background
def create_object_mask(image, dimensions):
    width = dimensions[0]
    height = dimensions[1]
    channels = dimensions[2]
    size = int(image.size / channels)
    summation = 0
    highest = 0
    for x in range(width):
        for y in range(height):
            intensity = sum(image[x,y]) / 3
            summation += intensity
            if intensity > highest:
                highest = intensity
    average = summation / size
    mask = numpy.zeros([width, height, 1], numpy.uint8)
    mask_cutoff = (average + highest) / 2
    for x in range(width):
        for y in range(height):
            intensity = sum(image[x,y]) / 3
            if intensity >= mask_cutoff:
                mask[x,y] = 255
    return mask
