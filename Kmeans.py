__authors__ = ['1669013','1671506', '1667730', '1672251']
__group__ = '11'

import numpy as np
import utils
import math
from random import randint

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
        Constructor of KMeans class
            Args:
                K (int): Number of cluster
                options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self.centroids = []
        self.percentatges = []


    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        if isinstance(X, list):
            X = np.array(X)
        X = X.astype(float)
        if len(X.shape) == 3 and X.shape[2] == 3:
            X = X.reshape(-1, 3)
        self.X = X


    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options


    def _init_centroids(self):
        """
        Initialization of centroids
        """

        if self.options['km_init'].lower() == 'first':
            self.old_centroids = np.copy(self.centroids)
            unique_values, indices = np.unique(self.X, axis=0, return_index=True)
            indices.sort()
            if len(indices) >= self.K:
                unique_points=self.X[indices[:self.K]]
                self.centroids = unique_points

        elif self.options['km_init'].lower() == 'random':
            self.old_centroids = self.centroids
            unique_points = []
            while len(unique_points) < self.K:
                point = self.X[randint(0, self.X.shape[0]-1)]
                if not any(np.array_equal(point, u) for u in unique_points):
                    unique_points.append(point)
            self.centroids = np.asfarray(np.array(unique_points))

        elif self.options['km_init'].lower() == 'custom':
            self.old_centroids = self.centroids
            unique_points = []
            i = 0
            while len(unique_points) < self.K:
                if not any(np.array_equal(self.X[i], u) for u in unique_points):
                    unique_points.append(self.X[i])
                i += int((self.X.shape[0]/self.K)-1)
                if i >= self.X.shape[0]:
                    i -= self.X.shape[0]
            self.centroids = np.asfarray(np.array(unique_points))
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])


    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        distances=distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1) # Calcular el index del valor mínim per filas

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids=self.centroids
        centroids=np.empty((self.K, self.X.shape[1]))
        for k in range(self.K):
            indices = np.where(self.labels == k)
            points=self.X[indices]
            mitjana=np.mean(points, axis=0)
            centroids[k]=mitjana
        self.centroids=np.asfarray(centroids)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        if np.array_equal(self.old_centroids, self.centroids):
            return True
        else: return False

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        
        while not self.converges() and self.num_iter < self.options['max_iter'] :
            self.get_labels() #Per a cada punt de la imatge, troba quin és el centroide més proper.
            self.get_centroids() #Calcula nous centroides utilitzant la funció get_centroids
            self.num_iter+=1 #Augmenta en 1 el nombre d’iteracions

    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """
        dist=distance(self.X,self.centroids)
        wcd=1/np.shape(self.X)[0] * np.sum(np.min(dist, axis=1)**2)
        return wcd

    def interClassDistance(self):
        """
        returns the inter class distance of the current clustering
        """
        icd = 0
        for k1 in range(self.K):
            for k2 in range(self.K):
                if k2!=k1:
                    c1 = self.X[self.labels == k1]
                    c2 = self.X[self.labels == k2]
                    icd += np.mean(np.power(c1[:,np.newaxis]-c2,2))
        return icd

    def fisherDiscriminant(self):
        icd = self.interClassDistance()
        wcd = self.withinClassDistance()
        return wcd/icd

    def find_bestK(self, max_K):
        """
        sets the best k analysing the results up to 'max_K' clusters
        """
        if self.options['fitting'] == 'WCD':
            self.__init__(self.X, 1, options=self.options)
            self.fit()
            old_wcd=self.withinClassDistance()
            for k in range(2, max_K + 1):
                self.__init__(self.X, k, options=self.options)
                self._init_options(self.options)
                self.fit()
                new_wcd=self.withinClassDistance()
                dec = 100*(new_wcd/old_wcd)
                old_wcd=new_wcd
                if abs(100-dec) < 20:
                    self.K = k-1
                    break
                
        elif self.options['fitting'] == 'ICD':
            self.__init__(self.X, 2)
            self.fit()
            old_icd=self.interClassDistance()
            for k in range(3, max_K + 1):
                self.__init__(self.X, k, options=self.options)
                self._init_options(self.options)
                self.fit()
                new_icd=self.interClassDistance()
                dec = 100*(old_icd/new_icd)
                old_icd=new_icd
                if abs(100-dec) < 20:
                    self.K = k-1
                    break
            
        elif self.options['fitting'] == 'Fisher':
            fd = [0] * (max_K -1)
            for k in range(2, max_K + 1):
                self.k = k
                self.__init__(self.X, k, options=self.options)
                self._init_options(self.options)
                self.fit()
                fd[k-2] = self.fisherDiscriminant()
            perc_dec = [100] * (len(fd))
            for i in range(1, len(fd)):
                perc_dec[i-1] =  fd[i] / fd[i - 1]
                if perc_dec[i-1] < 1:
                    self.K = i + 1
                    break
            else:
                self.K = max_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - C)**2, axis=2))
    return dist

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    labels = []
    percentatges = []
    probabs = utils.get_color_prob(centroids)
    for p in probabs:
        max_color = np.argmax(p)
        color = utils.colors[max_color]
        labels.append(color)
        percentatges.append(p[max_color])
    return labels, percentatges

