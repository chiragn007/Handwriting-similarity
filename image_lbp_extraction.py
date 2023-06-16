import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp_features(image):

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.reshape(-1, 1)  

    return hist