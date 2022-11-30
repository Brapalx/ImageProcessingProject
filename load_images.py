import rawpy
import numpy
import os

def load_image(filename, log):
    if log == True:
        print("Loaded file: " + filename)
    return rawpy.imread(filename).postprocess()


def load_image_with_prefix(prefix, log):
    return lambda fn : load_image(prefix + fn, log)


# Loads all of the images in a folder into an array of images (numpy arrays)
# Folder is opened rellative to executable path
# folder_name - name of the folder to open in the format of "name" or "./name", no trailing slash
# return = array of images (numpy arrays) from that folder
def load_folder(folder_name, log=False):
    filenames = os.listdir(folder_name)
    return list(map(load_image_with_prefix(folder_name + "/", log), filenames))
