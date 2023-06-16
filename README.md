## Handwriting-similarity

•In this code, the image_preprocess performs image resizing and conversion to grayscale. 
•The image_contour_extraction function uses contour detection to extract shape descriptors such as area and perimeter from the contours.
•The extract_texture_characteristics function utilizes the Histogram of Oriented Gradients (HOG) algorithm to extract texture characteristics from the image. 
•The image_similarity brings all the steps together, extracts features from the images, calculates similarity scores for shape descriptors, stroke features, and texture characteristics, and finally calculates the overall similarity score.
