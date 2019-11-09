import numpy as np
from scipy import fftpack
from skimage import util, filters
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.morphology import binary_dilation, binary_erosion, area_closing
from skimage.feature import canny
from helper_functions import show_images

#Function expects the image in grayscale
def extract(img):
    img[img[:,:] < 1] = 0
    img = util.invert(img)
    img = binary_dilation(img)
    s = generate_binary_structure(2,2)
    features, number = label(img, structure = s)
    out = np.copy(features)
    out[features[:,:] <= 0] = 255
    out[features[:,:] > 0] = 0
    #show_images([features, out],['Feature Photo', 'Number Of Detected Words:{}'.format(number)])
    return features, number
