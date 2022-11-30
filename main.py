from load_images import load_folder
from light_stack import stack_light_images
from flat_stack import stack_flat_images
from dark_stack import stack_dark_images
from object_detection import create_object_mask

import cv2 as cv


# Main function here
light_image_array = load_folder("./lights")
flat_image_array = load_folder("./flats")
dark_image_array = load_folder("./darks")
bias_image_array = load_folder("./biases")
dimensions = light_image_array[0].shape
master_light_image, new_dimensions = stack_light_images(light_image_array, dimensions)
master_flat_image = stack_flat_images(flat_image_array, dimensions)
master_dark_image = stack_dark_images(dark_image_array, dimensions, 2, 10)

# cv.imwrite("final_light_stack_image.png", master_light_image)