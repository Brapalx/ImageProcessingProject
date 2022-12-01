import cv2
import numpy as np

def boost_saturation(rgb_image, value):
    imghsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s * value
    s = np.clip(s, 0, 255)
    imghsv = cv2.merge([h, s, v])
    imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return imgrgb

