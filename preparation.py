# Import the pySIFT code.
import sys
sys.path.append("pySift")
from pySift import sift, matching

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import k_means

#
# Your goal here: Get SIFT features from training images and cluster them!
# Define the number of clusters.

# Load the file containing the training images.
trainimages = [line.strip().split(" ")[0] for
               line in open("trainset-overview.txt", "r")]
print "There are", len(trainimages), "training images"

def extract_sift(trainimage):
    sigma       = 1.0
    hespoints   = sift.computeHes(trainimage, sigma, magThreshold=15,
                                  hesThreshold=10, NMSneighborhood=10)
    harpoints   = sift.computeHar(trainimage, sigma, magThreshold=5,
                                  NMSneighborhood=10)
    allpoints = np.concatenate((hespoints, harpoints))
    point1, sift1 = sift.computeSIFTofPoints(trainimage, allpoints, sigma,
                                             nrOrientBins=8, nrSpatBins=4,
                                             nrPixPerBin=4)
    return point1, sift1

# Put the SIFT features from all training images in this variable.
trainpoints = []

for i in xrange(len(trainimages)):

# Extract point locations from the image using your selected point method
#and parameters.
# Compute the SIFT features.
    point1, sift1 = extract_sift(trainimages[i])
    trainpoints.extend(sift1)
# Cluster the SIFT features and put them in a matrix with the name 'clusters'!
print "Clustering..."

def cluster_data(features, k, nr_iter=25):
    centroids = k_means(features, n_clusters=k, max_iter=nr_iter)[0]
    return centroids
# Specify the number of clusters to use.
k = 100
# Cluster the synthetic data defined previously.
clusters = cluster_data(trainpoints, k)
print 'done!'

def euclidean_distance(x, y):
    assert(len(x) == len(y))
    return np.sum((x-y)**2)**.5

def distances(a,X,distance_fn=euclidean_distance):
#Return a list of distances between vector a, and each row of X
#USE distance_fn to calculate distances. eg: some_dist = distance_fn(a,b)
#We create an array to store the distances in
    dists = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        dist = distance_fn(a, X[i])
        dists[i] = dist
    return dists

# You can store the histogram results in the following:
# This is the length of your histogram vector.
size_of_histograms = k #equal to the number of clusters!
train_feat = np.zeros((len(trainimages), size_of_histograms))

for i, image in enumerate(trainimages):
    point1, image_sift = extract_sift(image)
    for point in image_sift:
        x = [euclidean_distance(point, clust) for clust in clusters]
        clusterin = np.argmin(x)
        train_feat[i][clusterin] += 1
print 'done!'
# Go through the SIFTs of every image and create a histogram for the image
# relative to the clusters you discovered in the previous phase.
