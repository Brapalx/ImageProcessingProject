from load_images import load_folder
from light_stack import stack_light_images
from flat_stack import stack_flat_images
from dark_stack import stack_dark_images
from bias_stack import stack_bias_images
from subtract_dark import subtract_dark
from subtract_flat import subtract_flat_frame
from subtract_bias import subtract_bias
from object_detection import create_object_mask
from boost_saturation import boost_saturation


import cv2 as cv


# Main function here

def image_stacking_all():
    light_image_array = load_folder("./lights")
    flat_image_array = load_folder("./flats")
    dark_image_array = load_folder("./darks")
    bias_image_array = load_folder("./biases")
    dimensions = light_image_array[0].shape
    master_light_image, new_dimensions = stack_light_images(light_image_array, dimensions)
    master_flat_image = stack_flat_images(flat_image_array, dimensions)
    master_dark_image = stack_dark_images(dark_image_array, dimensions, 2, 10)
    master_bias_image = stack_bias_images(bias_image_array, dimensions)
    cv.imwrite("final_light_stack_image.png", master_light_image)
    cv.imwrite("final_flats_stack_image.png", master_flat_image)
    cv.imwrite("final_dark_stack_image.png", master_dark_image)
    cv.imwrite("final_bias_stack_image.png", master_bias_image)

image_stacking_all()

light = cv.imread("final_light_stack_image.png")
dark = cv.imread("final_dark_stack_image.png")
flats = cv.imread("final_flats_stack_image.png")
bias = cv.imread("final_bias_stack_image.png")

sub_dark = subtract_dark(light, dark)
cv.imwrite("sub_dark_image.png", sub_dark)

sub_flat = subtract_flat_frame(sub_dark, flats)
cv.imwrite("sub_flat_image.png", sub_flat)

sub_bias = subtract_bias(sub_flat, bias)
cv.imwrite("sub_bias_image.png", sub_bias)

star_mask = create_object_mask(sub_bias, [3176, 4770, 3])
cv.imwrite("star_mask.png", star_mask)
masked_img = cv.bitwise_and(sub_bias, sub_bias, mask = star_mask)
cv.imwrite("masked_image.png", masked_img)

boost_sat = boost_saturation(masked_img, 2)
cv.imwrite("boost_sat_image.png", boost_sat)





