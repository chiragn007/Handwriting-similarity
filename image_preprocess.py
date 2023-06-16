from sklearn import preprocessing
import cv2

def preprocess(image):

    image = cv2.resize(image, (500, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray