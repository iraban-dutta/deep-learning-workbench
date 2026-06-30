from builtins import range
from builtins import object
import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0, use_matmul=True):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
        -use_matmul: flag to use matrix multiplication method when num_loops=0

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X, use_matmul)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return dists, self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                d_l2_ij = np.sqrt((np.abs(X[i]-self.X_train[j])**2).sum(axis=-1))
                dists[i][j] = d_l2_ij
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            
            temp = (X[i] - self.X_train) # Broadcast-> X[i]:(3072, ) | X_train:(5000, 3072) | temp:(5000, 3072)
            d_l2_i = np.sqrt((np.abs(temp)**2).sum(axis=-1))
            dists[i] = d_l2_i
        return dists

    def compute_distances_no_loops(self, X, use_matmul=True):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        if use_matmul:
            # Use Vectorized Expression: Efficient
            test_sq = (X**2).sum(axis=-1, keepdims=True) # X:(500, 3072) | test_sq:(500, 1) 
            train_sq = (self.X_train**2).sum(axis=-1)    # X_train:(5000, 3072) | train_sq:(5000, ) 
            temp1 = test_sq + train_sq                   # Broadcast-> test_sq:(500, 1) | train_sq:(5000, 1) | temp1:(500, 5000)
            temp2 = 2*np.matmul(X, self.X_train.T)       # X:(500, 3072) | X_train.T:(3072, 5000) | temp: (500, 5000)
            dists = np.sqrt(temp1-temp2)
        else: 
            # Using broadcasting with singleton dimension: Very High Memory Usage and inefficient
            test_e = np.expand_dims(X, axis=1)
            train_e = np.expand_dims(self.X_train, axis=0)
            temp = test_e - train_e # Broadcast-> X:(500, 1, 3072) | X_train:(1, 5000, 3072) | temp:(500, 5000, 3072)
            dists = np.sqrt((np.abs(temp)**2).sum(axis=-1))
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################

            # Pick the ith row of dists, sort and return idxs
            disti_sorted_idx = np.argsort(dists[i])
            # Lookup into training labels correspoinding to closest k neighbours
            closest_y = self.y_train[disti_sorted_idx][:k]


            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            freq = {}
            for j in range(k):
                freq[closest_y[j]] = freq.get(closest_y[j], 0) + 1

            best_label = None
            best_cnt = 0
            for label, cnt in freq.items():
                if cnt>best_cnt:
                    best_cnt=cnt
                    best_label=label
                elif cnt==best_cnt and label<best_label:
                    best_label=label


            y_pred[i] = best_label

        return y_pred
    

if __name__=="__main__":

    X_tr = np.random.rand(10, 3072)
    Y_tr = np.arange(10)

    X_ts = np.random.rand(2, 3072)
    Y_t2 = np.arange(2)

    knn_model = KNearestNeighbor()
    knn_model.train(X_tr, Y_tr)
    dist_2 = knn_model.predict(X_ts, 1, 2)
    dist_1 = knn_model.predict(X_ts, 1, 1)
    dist_0 = knn_model.predict(X_ts, 1, 0)
    print(dist_2.shape)
    print(dist_1.shape)
    print(dist_0.shape)

