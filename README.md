# FaceDetection

# Face Detection and Clustering

This repository contains Python code for face detection and clustering using OpenCV and face_recognition libraries.

## Introduction

This project focuses on detecting faces in images and clustering them based on their similarities. It includes two main functions:

1. `detect_faces(img: np.ndarray) -> List[List[float]]`: This function takes an input image and returns the bounding boxes of detected faces in the image.

2. `cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]`: This function clusters input images based on their detected faces using K-means clustering.

## Requirements

To run the code in this repository, you need to have the following libraries installed:

- OpenCV (cv2)
- NumPy
- face_recognition

## Usage

1. Clone the repository to your local machine using:

git clone https://github.com/your-username/face-detection-clustering.git

2. Install the required libraries using:

pip install opencv-python numpy face_recognition

3. Navigate to the repository folder and run `task1.py` for face detection and `task2.py` for face clustering.

4. Follow the instructions provided in the code comments to use the functions and explore the results.




