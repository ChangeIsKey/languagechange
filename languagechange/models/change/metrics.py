from scipy.spatial.distance import cdist, cosine
from languagechange.models.meaning.clustering import Clustering
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
from typing import List, Union

class ChangeModel():

    def __init__(self):
        pass


class BinaryChange(ChangeModel):

    def __init__(self):
        pass

    def predict(self):
        pass


class GradedChange(ChangeModel):

    def __init__(self):
        pass

    def compute_scores(vectors_list):
        pass


class Threshold(BinaryChange):

    def __init__(self):
        pass

    def set_threshold(self, threshold):
        self.threshold = threshold


class AutomaticThrehold(Threshold):

    def __init__(self):
        pass

    def compute_threshold(self, scores, func = lambda x: np.mean(x)):
        self.threshold = func(scores)


class OptimalThrehold(Threshold):

    def __init__(self):
        pass

    def compute_threshold(self, scores, vrange=np.arange(0.,1.), evaluator=None):
        best_score = None
        best_threshold = None

        for v in vrange:
            labels = np.array(scores < v, dtype=int)
            score = evaluator(labels)
            if score > best_score or best_score == None:
                best_score = score
                best_threshold = v

        self.threshold = best_threshold


class APD(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, metric='cosine'):

        return np.mean(cdist(embeddings1, embeddings2, metric=metric))


class PRT(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, metric='cosine'):

        return cosine(embeddings1.mean(axis=0), embeddings2.mean(axis=0))


class PJSD(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, clustering_algorithm, metric='cosine'):
        clustering = Clustering(clustering_algorithm)
        clustering.get_cluster_results(np.concatenate((embeddings1,embeddings2),axis=0))
        labels1 = clustering.labels[:len(embeddings1)]
        labels2 = clustering.labels[len(embeddings1):]
        return self.compute_scores_from_labels(labels1, labels2)

    def compute_scores_from_labels(self, labels1, labels2):
        all_labels = np.concatenate((labels1, labels2),axis=0)
        labels = set(all_labels)
        count1 = Counter(labels1)
        count2 = Counter(labels2)
        p,q = [], []
        for l in labels:
            if l in count1:
                p.append(count1[l]/len(all_labels))
            else:
                p.append(0.)
            if l in count2:
                q.append(count2[l]/len(all_labels))
            else:
                q.append(0.)

        return jensenshannon(p, q)