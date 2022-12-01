import rawpy
import numpy as np
import os
import cv2 as cv


def subtract_flat_frame(input_image_array , flat_image_array):
    subtractImage = np.subtract(flat_image_array , input_image_array)
    output4image = histogram_equalization(subtractImage)
    return output4image



def histogram_equalization(image_array):
    histogramArray = np.bincount(image_array.flatten(), minlength=256)
    numPixels = np.sum(histogramArray)
    histogramArray = histogramArray/numPixels

    cumulativeHistogramArray = np.cumsum(histogramArray);
    transformMap = np.floor(255 * cumulativeHistogramArray).astype(np.uint8)

    img_list = list(image_array.flatten())

    equalizationImageList = [transformMap[p] for p in img_list]
    eqImageArray = np.reshape(np.asarray(equalizationImageList), image_array.shape)
    return eqImageArray



def histogram_equalization_Colored(image_array):    
    blue,green,red = cv.split(image_array)
    #get histogram
    histo_b, bin_blue = np.histogram(blue.flatten(), 256, [0, 256])
    histo_g, bin_green = np.histogram(green.flatten(), 256, [0, 256])
    histo_r, bin_red = np.histogram(red.flatten(), 256, [0, 256])
    
    #cumulative sum
    cdfBlue = np.cumsum(histo_b)  
    cdfGreen = np.cumsum(histo_g)
    cdfRed = np.cumsum(histo_r)

    cdf_mask_blue = np.ma.masked_equal(cdfBlue,0)
    cdf_mask_blue = (cdf_mask_blue - cdf_mask_blue.min())*255/(cdf_mask_blue.max()-cdf_mask_blue.min())
    cdf_final_blue = np.ma.filled(cdf_mask_blue,0).astype('uint8')
  
    cdf_mask_green = np.ma.masked_equal(cdfGreen,0)
    cdf_mask_green = (cdf_mask_green - cdf_mask_green.min())*255/(cdf_mask_green.max()-cdf_mask_green.min())
    cdf_final_green = np.ma.filled(cdf_mask_green,0).astype('uint8')
    
    cdf_mask_red = np.ma.masked_equal(cdfRed,0)
    cdf_mask_red = (cdf_mask_red - cdf_mask_red.min())*255/(cdf_mask_red.max()-cdf_mask_red.min())
    cdf_final_red = np.ma.filled(cdf_mask_red,0).astype('uint8')

    # merge the images three channel
    img_blue = cdf_final_blue[blue]
    img_green = cdf_final_green[green]
    img_red = cdf_final_red[red]
  
    img_out = cv.merge((img_blue, img_green, img_red))

    return img_out



#example to histogram equalization
#input_image = cv.imread("MasterDark_ISO200_30s.tif")
#input_image2 = cv.imread("MasterFlat_ISO100.tif")

#out_image = subtract_flat_frame(input_image,input_image2)



# np_input_image = np.asarray(input_image)
# # np_input_image2 = np.asarray(input_image2)

# # out_image = subtract_flat_frame(np_input_image,np_input_image2)
# # out_image = histogram_equalization(np_input_image)
# out_image = histogram_equalization_Colored(np_input_image)

#cv.imwrite("outimage1.png", out_image)