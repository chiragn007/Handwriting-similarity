import numpy as np
from sklearn import preprocessing

def calculate_similarity(features1, features2):

    features1 = np.array(features1)
    features2 = np.array(features2)
    scaler = preprocessing.MinMaxScaler()
    features1_scaled = scaler.fit_transform(features1)
    features2_scaled = scaler.transform(features2)

    distance = np.linalg.norm(features1_scaled - features2_scaled)
    
    similarity_score = 1 / (1 + distance)

    return similarity_score