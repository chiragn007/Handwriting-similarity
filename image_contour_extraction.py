import cv2
from skimage.feature import local_binary_pattern


def extract_contour_features(image):

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_features = []
    for contour in contours:

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h


        contour_features.append((area, perimeter, aspect_ratio))

    return contour_features