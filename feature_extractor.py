import skimage.io as io
import numpy as np
from scipy import fftpack
from skimage.color import rgb2gray
from skimage import util, filters
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage.morphology import binary_dilation, binary_erosion, area_closing
from skimage.feature import canny


def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 


img = io.imread('test.png')
img = rgb2gray(img)
img[img[:,:] < 1] = 0
img = util.invert(img)
img = binary_dilation(img)
s = generate_binary_structure(2,2)
features, number = label(img, structure = s)
out = np.copy(features)
out[features[:,:] <= 0] = 255
out[features[:,:] > 0] = 0
show_images([features, out],['Feature Photo', 'Number Of Detected Words:{}'.format(number)])

