import rawpy
import numpy
import os

def load_image(filename):
    return rawpy.imread(filename).postprocess()


def load_image_with_prefix(prefix):
    return lambda fn : load_image(prefix + fn)


# Loads all of the images in a folder into an array of images (numpy arrays)
# Folder is opened rellative to executable path
# folder_name - name of the folder to open in the format of "name" or "./name", no trailing slash
# return = array of images (numpy arrays) from that folder
def load_folder(folder_name):
    filenames = os.listdir(folder_name)
    return list(map(load_image_with_prefix(folder_name + "/"), filenames))
