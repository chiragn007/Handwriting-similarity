import cv2
import numpy as np
from sklearn import preprocessing
from skimage.feature import local_binary_pattern
from image_preprocess import preprocess
from image_contour_extraction import extract_contour_features
from image_similarity import calculate_similarity
from image_lbp_extraction import extract_lbp_features
from image_hog_extraction import extract_hog_features


image1 = cv2.imread("Data/21.png")
image2 = cv2.imread("Data/53.png")

preprocessed_image1 = preprocess(image1)
preprocessed_image2 = preprocess(image2)

contour_features1 = extract_contour_features(preprocessed_image1)
contour_features2 = extract_contour_features(preprocessed_image2)
contour_similarity = calculate_similarity(contour_features1, contour_features2)

lbp_features1 = extract_lbp_features(preprocessed_image1)
lbp_features2 = extract_lbp_features(preprocessed_image2)
lbp_similarity = calculate_similarity(lbp_features1, lbp_features2)

hog_features1 = extract_hog_features(preprocessed_image1)
hog_features2 = extract_hog_features(preprocessed_image2)
hog_similarity = calculate_similarity(hog_features1, hog_features2)

print("similarity score:", lbp_similarity*100)
