import cv2
from sklearn import preprocessing

def extract_hog_features(image):

    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(image)
    hog_features = hog_features.reshape(-1, 1)

    return hog_features