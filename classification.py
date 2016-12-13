import numpy as np
import sys
sys.path.append("pySift")
from pySift import sift, matching
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#loading train and validation preprocessed data
train_feat = np.load('train_feat_big.npy')
val_feat = np.load('val_feat_big.npy')

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
    """Nearest Neighbour classification as a start to test classification."""
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
print "NN-classification Accuracy:", np.sum(pred_y == val_labels)/np.float_(
    len(val_labels))

scores = []
neighbours =[5,10,15,20,25,30,35,40,45,50,100]
print "K-Nearest-neighbour classification:"
for i in neighbours:
    nbrs = KNeighborsClassifier(n_neighbors=i).fit(train_feat, train_labels)
    scores.append(nbrs.score(val_feat, val_labels))
print "best scoring K: ", neighbours[np.argmax(scores)]
print "score: ", scores[np.argmax(scores)]

supvc = SVC(C=5.0, kernel='linear', decision_function_shape='ovr').fit(
    train_feat, train_labels)
print "SVC score: ", supvc.score(val_feat, val_labels)

mlp = MLPClassifier(solver='adam',activation='logistic',max_iter=1000,
                    alpha=10).fit(train_feat, train_labels)
print "MLP score: ", mlp.score(val_feat, val_labels)
