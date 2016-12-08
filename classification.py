import numpy as np
import sys
sys.path.append("pySift")
from pySift import sift, matching
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#loading train and validation preprocessed data
train_feat = np.load('train_feat_first.npy')
val_feat = np.load('val_feat_first.npy')

#Training and validation ground truth labels. Needed for classification
train_labels = np.array([int(line.strip().split(" ")[1]) for
                         line in open("trainset-overview.txt", "r")])
val_labels = np.array([int(line.rstrip().split(' ')[1]) for
                       line in open('valset-overview.txt','r')])

"""Simple function to calculate Euclidean distance"""
def euclidean_distance(x, y):
    assert(len(x) == len(y))
    return np.sum((x-y)**2)**.5

"""Simple function which will calculate the distances between a single image,
'a', and the X matrix of images."""
def distances(a,X,distance_fn=euclidean_distance):
#Return a list of distances between vector a, and each row of X
#USE distance_fn to calculate distances. eg: some_dist = distance_fn(a,b)
#We create an array to store the distances in
    dists = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        dist = distance_fn(a, X[i])
        dists[i] = dist
    return dists

def nn_classifier(test_X, train_X, train_y):
#We create an array for you to populate with your class predictions
#Go through each sample in test_X and predict its class
#based on the label of its nearest neighbor in train_X.
#Insert this prediction in 'predictions'
#(Use Euclidean Distance as your distance metric here)
    predictions = np.zeros(test_X.shape[0])

    for i, row in enumerate(test_X):
        hold = distances(row, train_X)
        predict = train_y[np.argmin(hold)]
        predictions[i] = predict
    return predictions

#Evaluate the quality of your model's predictions
pred_y = nn_classifier(val_feat, train_feat, train_labels)
print "Accuracy:", np.sum(pred_y == val_labels)/np.float_(len(val_labels))
