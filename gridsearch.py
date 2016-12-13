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
from sklearn.grid_search import GridSearchCV

#loading train and validation preprocessed data
X_train = np.load('train_feat_big.npy')
X_test = np.load('val_feat_big.npy')

#Training and validation ground truth labels. Needed for classification
y_train = np.array([int(line.strip().split(" ")[1]) for
                         line in open("trainset-overview.txt", "r")])
y_test = np.array([int(line.rstrip().split(' ')[1]) for
                       line in open('valset-overview.txt','r')])

knn_grid = [{'n_neighbors': [1,2,5,10,15,20,25],
             'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
             'leaf_size': [10,20,30,50],
             'metric': ['minkowski'],
             'p': [1,2]}] #1 = manhattan distance, 2 = euclidean

svc_grid = [{'C': [0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.3, 3, 30, 300,
                   3000],
             'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
             'degree':[1,3,10],
             'decision_function_shape': ['ovo', 'ovr', None]}]

mlp_grid = [{'hidden_layer_sizes': [(10,), (50,), (100,), (300,)],
             'activation': ['identify', 'logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
             'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10,
                       100]}]

knn_search = GridSearchCV(KNeighborsClassifier(), param_grid=knn_grid,
                          n_jobs=2, refit=True, error_score=0)
svc_search = GridSearchCV(SVC(), param_grid=svc_grid,
                          n_jobs=2, refit=True, error_score=0)
mlp_search = GridSearchCV(MLPClassifier(), param_grid=mlp_grid,
                          n_jobs=2, refit=True, error_score=0)


def report(grid_scores, n_top=3):
    results.write(
        'The top ' + str(n_top) + ' performing parameter settings are: \n')
    print 'The top ', n_top, ' performing parameter settings are: \n'
    topscores = sorted(grid_scores, key=getkey, reverse=True)[0:n_top]
    for i in topscores:
        results.write(str(i) + '\n')
        print i, '\n'
results=open('results.txt', 'w+')
for i in [knn_search, svc_search, mlp_search]:
    print 'start training'
    results.write('start training \n')
    start = time.time()
    i.fit(X_train, y_train)
    results.write('Done training, it took me: ' + str((time.time() - start))+' seconds \n')
    print 'done training'
    start = time.time()
    results.write(str(i) + ' score is: ' + str(i.score(X_test,y_test)) + "\n")
    print 'wrote scores'
    results.write(str('Made a prediction, it took me: ' + str((time.time() - start)) + ' seconds \n'))
    results.write("Residual sum of squares: %.2f" % np.mean((i.predict(X_test) - y_test) ** 2))
    print 'wrote sum of squares'
    report(i.grid_scores_, n_top=5)
    joblib.dump(i, dict[i])
results.close()
