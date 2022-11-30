import cv2 as cv
import numpy as np
from load_images import load_folder


def new_dimensions(addition_map, dimensions):
    top_left = [0,0]
    bottom_right = dimensions
    max = 0
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            if addition_map[x,y] > max:
                max = addition_map[x,y]
                top_left = [x,y]
            if addition_map[x,y] == max:
                bottom_right = [x,y]
    return [top_left, bottom_right]


def matching_homography_matrix(base, other):
    # Finds the homography matrix between the base and other image
    det = cv.ORB_create(nfeatures=100)
    kp1, desc1 = det.detectAndCompute(other, None)
    kp2, desc2 = det.detectAndCompute(base, None)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    source_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destin_points = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(source_points, destin_points, cv.RANSAC, 10.0)
    if mask.sum() < 10:
        return np.identity(3)
    return M

# Rotates and stacks light images together,
# also returns the new dimensions of the stacked image
# light_images - array of flat images (numpy arrays)
# dimensions - array of [ width , height , # of color channels ]
# Returns the stacked light images, the new top left and bottom right corners
def stack_light_images(light_images, dimensions):
    middle = int(len(light_images) / 2)
    counter_dimensions = [dimensions[0], dimensions[1], 1]
    width = dimensions[0]
    height = dimensions[1]
    base_img = light_images[middle]
    output_img = np.zeros(dimensions, int)
    counter_img = np.zeros(counter_dimensions, dtype=int)
    for img in light_images:
        M = matching_homography_matrix(base_img, img)
        output_img += cv.warpPerspective(img, M, (height, width))
        # counter = np.ones(counter_dimensions, dtype=int)
        # ounter_img += cv.warpPerspective(counter, M, (width, height))
    return ((output_img*255)/output_img.max()).astype('uint8'), new_dimensions(counter_img, [width, height])
