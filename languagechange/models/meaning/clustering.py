import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import networkx
# TEMPORARY import, needs a proper installation from requirements.txt
from correlation import cluster_correlation_search


class ClusteringResults():
    def __init__(self, labels):
        self.labels = labels
        self.cluster2instances = {}
        for j,l in enumerate(labels):
            if not l in self.cluster2instances:
                self.cluster2instances[l] = []
            self.cluster2instances[l].append(j)

    def get_cluster_instances(self, cluster_id):
        return self.cluster2instances[cluster_id]


class Clustering():
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def get_cluster_results(self, examples):
        self.labels = self.algorithm.fit_predict(examples)
        return ClusteringResults(self.labels)
    
# The class below is a modified version of the class in https://github.com/FrancescoPeriti/WiDiD/blob/main/src/cluster.py
# Author: Francesco Periti francesco.periti@unimi.it
class APosterioriaffinityPropagation(ClusterMixin, BaseEstimator):
    """A class that implements the APP clustering algorithm.
    This class is compatible with the [scikit-learn](https://scikit-learn.org) ecosystem.
    Parameters
    ----------
    damping : float, default=0.9
        Damping factor in the range `[0.5, 1.0)` is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).
    max_iter : int, default=200
        Maximum number of iterations.
    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.
    copy : bool, default=True
        Make a copy of input data.
    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.
    affinity : {'euclidean', 'cosine'}, default='cosine'
        Which affinity to use. At the moment ``cosine``,
        ``euclidean`` are supported. 'euclidean' uses the
        negative squared euclidean distance between points.
    verbose : bool, default=False
        Whether to be verbose.
    random_state : int, RandomState instance or None, default=42
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
    th_gamma : int, default=0
        Threshold over the aging index gamma. Must be in [1, ∞).
        Clustering refinement is not enforced when th_gamma=0.
    Attributes
    ----------
    step_ : int
        Iteration number.
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of cluster centers.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.
    n_iter_ : int
        Number of iterations taken to converge.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    memory_ : dict
        Memory of all clustering result.
    """

    def __init__(self, affinity: str = 'cosine',
                 damping: float = 0.9, max_iter: int = 200,
                 convergence_iter: int = 15, copy: bool = True, preference: bool = None,
                 verbose: bool = False, random_state: int = 42, th_gamma: int = 0,
                 pack='centroid', singleton='one'):

        self._ap = AffinityPropagation(damping=damping, max_iter=max_iter,
                                       convergence_iter=convergence_iter, copy=copy,
                                       verbose=verbose, preference=preference,
                                       random_state=random_state, affinity='precomputed')
        self.affinity = affinity
        self.th_gamma = th_gamma

        self.step_ = 0
        self.memory_ = dict()
        self.pack_ = pack
        self.singleton_ = singleton

    def _affinity(self, X: np.array, Y: np.array = None) -> np.array:
        '''
        Compute the matrix of similarity between X and Y
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        Y : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster, or exemplars of previously clustered instances.
        Returns
        -------
        affinity_matrix : ndarray of shape (n_samples,)
            Affinity matrix.
        '''
        if self.affinity == 'cosine':
            return cosine_similarity(X, Y)
        elif self.affinity == 'euclidean':
            return -euclidean_distances(X, Y, squared=True)

    def _pack(self, X):
        '''
        Pack clusters of vectors into single representations
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Clustered training instances.
        Returns
        -------
        cluster_centers_ : ndarray of shape (n_clusters,)
            Indices of cluster centers that act as cluster exemplars
        '''

        unique_labels = np.sort(np.unique(self._ap.labels_))
        record = {self.step_: {l: np.where(self._ap.labels_ == l)[0] for l in unique_labels}}
        self.memory_.update(record)

        # -- centroid --
        if self.pack_ == 'centroid':
            return X[self._ap.cluster_centers_indices_]

        # -- mean --
        if self.pack_ == 'mean':
            labels = np.unique(self._ap.labels_)
            means = np.array([X[self._ap.labels_ == l].mean(0) for l in labels])
            return means

        # -- most similar --
        if self.pack_ == 'most_similar':
            labels = np.unique(self._ap.labels_)
            means = np.array([X[self._ap.labels_ == l].mean(0) for l in labels])
            exemplar_indices = [self._affinity(X[self._ap.labels_ == l], [means[i]]).argmax()
                                for i, l in enumerate(labels)]
            return np.array([X[self._ap.labels_ == l][idx]
                             for l, idx in zip(labels, exemplar_indices)])
        
    # This method is not in the priginal package by Francesco Periti
    def fit_predict(self, embs : [np.array]): # embs is a list of embeddings
        for emb in embs:
            self.fit(emb)
        return self.labels_

    def fit(self, X, y=None):
        """
        Fit the clustering from features.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self
            Returns the instance itself.
        """

        if self._ap.verbose:
            print(f"--- {self.step_}-th step ---")

        if self.step_ == 0:
            curr_X = X
        #elif not hasattr(self, 'cluster_centers_'):
        #    curr_X = X
        else:
            curr_X = np.concatenate([self.cluster_centers_, X])

        self.affinity_matrix_ = self._affinity(curr_X)
        self._ap.fit(self.affinity_matrix_)

        if np.unique(self._ap.labels_).shape[0] == 1:
            n = curr_X.shape[0]

            # -- all
            if self.singleton_ == 'all':
                self._ap.labels_ = np.arange(0, n)
                self._ap.cluster_centers_indices_ = np.arange(0, n)
                if self._ap.verbose:
                    print(f'{X.shape[0]} singleton clusters created')

            # one
            if self.singleton_ == 'one':
                self._ap.labels_ = np.zeros((n))
                self._ap.cluster_centers_indices_ = np.array([0])
                if self._ap.verbose:
                    print('One singleton clusters created')

        self.cluster_centers_ = self._pack(curr_X)
        self._refinement()
        self.step_ += 1
        self.labels_ = self._unpack()

    def _unpack(self):
        '''Unpack the cluster representations from self.memory_ into a single label array.'''

        tot = self._n_objects()
        labels = -np.ones((tot))
        matches = self._mapping_history()

        for i in range(0, self.step_):
            if i == 0:
                for k in self.memory_[i]:
                    members = self.memory_[i][k]
                    labels[members] = k
                continue

            # perform mapping
            match = matches[i]
            for k, v in match.items():
                labels[labels == k] = v

            # append new labels to the array
            n = labels[labels != -1].shape[0]
            exemplars = self.memory_[i - 1]
            n_exemplars = len(exemplars)
            offset = n - n_exemplars

            for k in self.memory_[i]:
                members = self.memory_[i][k]
                new_values = [m for m in members if m not in exemplars]
                new_values = np.array(new_values) + offset
                if len(new_values) > 0:
                    labels[new_values] = k

        return labels

    def _mapping_history(self):
        """
        Creates a mapping between the memory of two consecutive steps.
        For each step, the method generates a dictionary with keys being the elements in the memory of the current step and
        values being the corresponding element in the memory of the next step. The resulting mapping history is returned as
        a dictionary with step number as keys and the step's mapping as values.
        Returns
        -------
            matches : dict
            A dictionary with step numbers as keys and step's mappings as values
        Note
        ----
        The mapping history generated is used to determine the relationship
        between memory across consecutive steps.
        """

        # if self.step_ == 0:
        #    return {self.step_: {k: k for k in self.memory_[self.step_]}}

        matches = dict()
        for step in range(0, self.step_ - 1):
            match = dict()
            for k in self.memory_[step].keys():
                for k1 in self.memory_[step + 1].keys():
                    if k in self.memory_[step + 1][k1]:
                        match[k] = k1
            matches.update({step + 1: match})
        return matches

    def _refinement(self):
        """
        Refines the clustering result by checking for cluster updates and removing aged clusters.
        Aged clusters are those that are not updated for more than 'th_gamma' steps.
        Parameters
        ----------
        th_gamma : int
            Clustering refinement threshold.
            If the th_gamma parameter is 0, no clustering refinement is enforced.
        """

        if self.th_gamma == 0:
            return

        for k in self._ap.labels_:
            if self._last_update(k) >= self.th_gamma:
                self._forgot(k)

    def _last_update(self, k, step=None):
        '''
        Returns the number of steps since the last update of the cluster k.
        This method calculates the number of steps since the last update of the cluster k by
        recursively checking the previous steps in the self.memory_ dictionary. If the cluster
        k is found in a previous step with only one member, the count of step without updates is incremented and
        the search continues.
        Parameters
        ----------
        k : int
            Cluster label to search for.
        step : int, optional, deafult=None
            Step number in the memory to start the search from.
            When 'step' is None, the search starts from self.step_
        Returns
        -------
            last_update : int
            The number of steps since the last update of the cluster k.
        '''
        if step is None:
            step = self.step_

        if step < 0 or k not in self.memory_[step] or len(self.memory_[step][k]) > 1:
            return 0
        elif step >= 0 and len(self.memory_[step][k]) == 1:
            return 1 + self._last_update(self.memory_[step][k][0], step - 1)

    def _forgot(self, k, step=None):
        """
        This method implements the forget process, where a cluster k is deleted from memory.
        The cluster is recursively deleted in previous steps until all the related information is erased from memory.
        Parameters
        ----------
        k : int
            Cluster to delete.
        step : int, optional, deafult=None
            Step number in the memory to start the delete process.
            When 'step' is None, the process starts from self.step_.
            When the step is 0, it directly deletes k from memory.
            When k is present in the memory of the given step, it will
            recursively delete the corresponding members in previous
            steps until all the related information to k is erased from memory.
        """

        if step is None:
            step = self.step_

        if step == 0 and k in self.memory_[step]:
            del self.memory_[step][k]

        if k in self.memory_[step]:
            for v in self.memory_[step][k][:]:
                self._forgot(v, step - 1)
            del self.memory_[step][k]

    def _n_objects(self):
        """
        Calculates the total number of objects in all memory steps.
        The method starts by counting the number of objects in the first step's memory.
        Then it iterates through each subsequent step, adding the number of objects in
        each step that are not already in previous steps.
        Returns
        -------
            tot : int
                The total number of objects in all memory steps.
        """
        tot = sum([len(values) for values in self.memory_[0].values()])
        for i in range(1, self.step_):
            for k in self.memory_[i].keys():
                values = [v for v in self.memory_[i][k] if v not in self.memory_[i - 1].keys()]
                tot += len(values)
        return tot
    

class CorrelationClustering():
    def __init__(self, s = 10, max_attempts = 200, max_iters = 5000, initial = [], split_flag = True):
        self.s = s
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.initial = initial
        self.split_flag = split_flag

    def classes_to_node2cluster(self, classes):
        """
        classes: list of sets, each set is a cluster of node IDs.
        returns: dict node -> cluster_index
        """
        node2cluster = {}
        for c_idx, cluster in enumerate(classes):
            for node in cluster:
                node2cluster[node] = c_idx
        return node2cluster

    def fit_predict(self, similarity_graph : networkx.classes.graph.Graph):
        """
            Args:
                similarity_graph (networkx.classes.graph.Graph) : a graph where each node is the id of a usage and each edge is a similarity between two usages (>0 for positive edges and <0 for negative edges)
        """
        for init in range(5):
            classes, _ = cluster_correlation_search(
                similarity_graph, 
                s=self.s,
                max_attempts=self.max_attempts,
                max_iters=self.max_iters,
                initial=self.initial,
                split_flag=self.split_flag
            )
        node2cluster = self.classes_to_node2cluster(classes)
        labels = [node2cluster[n] for n in list(similarity_graph.nodes)]
        return labels