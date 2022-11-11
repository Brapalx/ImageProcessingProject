from load_images import load_folder
from flat_stack import stack_flat_images
from object_detection import create_object_mask

import cv2 as cv


# Main function here
flat_image_array = load_folder("./flats")
master_flat_image = stack_flat_images(flat_image_array, flat_image_array[0].shape)
