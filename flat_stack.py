import cv2
import numpy


def calculate_data_point_with_prefix(images, num_images, output):
    return lambda coords : calculate_data_point(images, num_images, output, coords[0], coords[1])

def calculate_data_point(images, num_images, output, x, y):
    data_set = numpy.zeros([num_images, 2], tuple)
    for i in range(num_images):
        dot_p = numpy.array([1, 0, 0]).dot(images[i][x, y])
        data_set[i] = [dot_p, i]
    data_set = numpy.sort(data_set, axis=0)
    output[x, y] = images[data_set[int(num_images / 2)][1]][x, y]

# Uses median to stack all of the flat images
# flat_images - array of flat images (numpy arrays)
# dimensions - array of [ width , height , # of color channels ]
# return = a single merged flat image
def stack_flat_images(flat_images, dimensions):
    return numpy.median(flat_images, axis=0)
    width = dimensions[0]
    height = dimensions[1]
    channels = dimensions[2]
    master_flat = numpy.zeros([width, height, channels])
    coordinates = []
    for x in range(width):
        for y in range(height):
            coordinates.append((x, y))
    list(map(calculate_data_point_with_prefix(flat_images, len(flat_images), master_flat), coordinates))
    return master_flat