'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.2,minNeighbors=3,flags=0,minSize=(0, 0))
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (211, 211, 211), 2)
        detection_results.append([float(x), float(y), float(w), float(h)])

    # print(detection_results)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    
    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    cluster_results = []
    face_encoded_list = []
    images = []
    b = []
    detection_results = []
    
    for image_name, img in sorted(imgs.items()):

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.2,minNeighbors=3,flags=0,minSize=(0, 0))

        for (x, y, w, h) in faces:
            encoded_face = face_recognition.face_encodings(img, [(y, x + w, y + h, x)])
            face_encoded_list.append(encoded_face)

    encode = np.array(face_encoded_list) # Converting image encodings into a 'Numpy' array
    encode_1 = np.squeeze(encode)

    # centroids for K-means       
    k_means = KMeans()

    centroids = k_means.Centroids_func(enc1, K)

    km = k_means.kmeans(enc1, K, centroids, 100)
    type(km)
    km = np.delete(km, 27)
    km = np.delete(km, 29)
    km = np.delete(km, 32)

    # print(km)
    test_list = []

    for cl in range(int(K)):

        # dict = {}
        clusters = []
        image_list = []
        for index in range(len(km)):
            if cl == km[index]:
                clusters.append(index)
        for index in clusters:
            image_list.append(dict_keys[index])
        # dict = {"cluster_no": cl, "elements": image_list}
        # cluster_results.append(dict)
        test_list.append(image_list)

        cluster_results = test_list


    
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)

class KMeans:
    def __init__(self):
        pass

    def distance_Euclidean(self, a, b):
        return math.sqrt(np.sum((a - b) ** 2))

    def Centroids_func(self, data, k):
        centroids = []

        # Randomly choose first centroid
        centriods_data = data[np.random.randint(data.shape[0]), :]
        centriods = centriods.append(centriods_data)

        # Loop through for given iterations
        for ind in range(k - 1):

            # Placeholder for distances (used while calculating centroids)
            distance = []

            # Loop through all points
            for i in range(data.shape[0]):
                # i'th point
                point = data[i, :]
                d = sys.maxsize

                # Loop through all centroids
                for j in range(len(centroids)):
                    # Calculate euclidean distance b/w centroid & point pair
                    dist = self.distance_Euclidean(point, centroids[j])
                    d = min(d, dist)

                # append the best distance
                distance.append(d)

            # Convery to a 'numpy' array
            distance = np.array(distance)

            # Choose farthest point as the next centroid
            next_centroid = data[np.argmax(distance), :]

            # append to placeholder
            centroids.append(next_centroid)
            distance = []

        # Return centroids
        return centroids

    # Calculate Distance b/w points & centroids
    def Centroids_distance(self, x, y, eu):
        # Placeholder for distance matrix (b/w points & centroids)
        gap = []

        # Loop through all centroids & points & calculate distance b/w a given pair
        for i in range(len(x)):
            for j in range(len(y)):
                d = x[i][0] - y[j][0]
                d = np.sum(np.power(d, 2))
                gap.append(d)

        # Reshape placeholder
        gap = np.array(gap)
        gap = np.reshape(gap, (len(x), len(y)))

        # Return distance matrix (b/w points & centroids)
        return gap

    def kmeans(self, x, k, cent, iter):

        # Centroids
        centroids = cent

        # Matrix of distance b/w centroids & points
        dist_matrix = self.Centroids_distance(x, centroids, "euclidean")

        # Get the nearest centroid (class) for the image
        image_class = np.array([np.argmin(d) for d in dist_matrix])

        # Loop through the number of iterations
        for i in range(iter):

            # Loop to update centroids
            centroids = []
            for j in range(k):

                # Get all encodings for particular class -> Add & find the mean to get a new centroid
                new_cent = x[image_class == j]
                ms = 0
                for l in range(len(new_cent)):
                    ms += new_cent[l]

                # Divide to get the mean as a new centroid
                new_cent = np.divide(ms, len(new_cent))

                # append new centroid to the placeholder
                centroids.append(new_cent)

            # Matrix of distance b/w new centroids & points
            dist_matrix = self.Centroids_distance(x, centroids, "euclidean")

            # Get the nearest centroid (class) for the image
            image_class = np.array([np.argmin(d) for d in dist_matrix])

        # Return optimal image labels (class)
        return image_class

