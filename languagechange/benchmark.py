from languagechange.resource_manager import LanguageChange
from languagechange.corpora import LinebyLineCorpus
from languagechange.models.representation.contextualized import ContextualizedModel
from languagechange.models.representation.definition import DefinitionGenerator
from languagechange.models.representation.prompting import PromptModel
from languagechange.usages import Target, TargetUsage, TargetUsageList, DWUGUsage
from languagechange.utils import NumericalTime, LiteralTime
from languagechange.models.meaning.clustering import Clustering, CorrelationClustering
import webbrowser
import os
import pickle
import json
from itertools import islice
import logging
import re
import zipfile
import random
import numpy as np
import math
import csv
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
import lxml.etree as ET
from typing import List, Dict, Union, Callable, Tuple
import inspect
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors


def purity(labels_true, cluster_labels):
    assert len(labels_true) == len(cluster_labels)
    N = len(labels_true)

    # Count the amount of usages of each label in each cluster.
    gold_labeled_clusters = {c : {} for c in set(cluster_labels)}
    for i in range(N):
        if labels_true[i] not in gold_labeled_clusters[cluster_labels[i]]:
            gold_labeled_clusters[cluster_labels[i]][labels_true[i]] = 0
        gold_labeled_clusters[cluster_labels[i]][labels_true[i]] += 1

    # Select the majority label of each cluster and save its count.
    majority_counts = [max(gold_labeled_clusters[c].values()) if len(gold_labeled_clusters[c]) != 0 else 0 for c in gold_labeled_clusters.keys()]
    # The purity is the sum of these counts divided by the total amount of samples.
    return sum(majority_counts) / N


class Benchmark():

    def __init__(self):
        pass

    def get_dataset(self, key):
        if key in self.data.keys():
            return self.data[key]
        else:
            raise KeyError
    
    def get_train(self):
        if 'train' in self.data.keys():
            return self.data['train']
        else:
            logging.info('Did not find a train set. Returning None')
    
    def get_dev(self):
        if 'dev' in self.data.keys():
            return self.data['dev']
        else:
            logging.info('Did not find a dev set. Returning None')
    
    def get_test(self):
        if 'test' in self.data.keys():
            return self.data['test']
        else:
            logging.info('Did not find a test set. Returning None')
    
    def get_all_data(self):
        if 'all' in self.data.keys():
            return self.data['all']
        else:
            all_data = []
            for k in set(self.data.keys()).difference({'all'}):
                all_data += self.data[k]
            return all_data
        
    def split_train_dev_test(self, train_prop = 0.8, dev_prop = 0.1, test_prop = 0.1, shuffle = True, epsilon = 1e-6):
        for s in ['train','dev','test']:
            if s in self.data.keys():
                logging.info(f'Dataset already contains a {s} set.')
        if not 'all' in self.data.keys():
            data = []
            for k in set(self.data.keys()).difference({'all'}):
                data.extend(self.data[k])
        else:
            data = self.data['all']
        
        try:
            assert abs(1 - (train_prop + dev_prop + test_prop)) < epsilon
        except AssertionError:
            logging.error('Train, dev and test proportions must add upp to 1.')
            return None

        if shuffle:
            data = random.sample(data, len(data))
            
        train_offset = int(len(data) * train_prop)
        dev_offset = int(len(data) * (train_prop + dev_prop))

        self.data['train'] = data[:train_offset]
        self.data['dev'] = data[train_offset:dev_offset]
        self.data['test'] = data[dev_offset:]

        for s in {'train','dev','test'}:
            if self.data[s] == []:
                del self.data[s]

    def get_data_by_word(self, dataset, word):
        dataset = self.get_dataset(dataset)
        return list(filter(lambda d : d['word'] == word, dataset))
    
    # Utility function which gets the character offsets from a tokenized sentence string and an index of the word in question.
    def word_index_to_char_indices(self, text, word_index, split_text = False):
        if split_text:
            text = text.split(" ")
        start = sum(len(s)+1 for s in text[:word_index])
        end = start + len(text[word_index])
        return start, end


class SemanticChangeEvaluationDataset(Benchmark):
    def __init__(self, dataset = None, language = None, version = None, name = None):
        self.dataset = dataset
        self.language = language
        self.version = version
        self.name = name

    def load_from_target_usages(self, target_usages : Dict[str, List[TargetUsage]], scores):
        self.target_words = target_usages.keys()
        self.target_usages_t1 = {}
        self.target_usages_t2 = {}
        for word, tu in target_usages.items():
            tu1, tu2 = tu
            self.target_usages_t1[word] = TargetUsageList(tu1)
            self.target_usages_t2[word] = TargetUsageList(tu2)

        
        if sum([int(type(s) == int and s == 1 or s == 0) for s in scores.values()]) == len(scores.values()):
            self.binary_task = {Target(word): score for word, score in scores.items()}
            self.graded_task = {}
        else:
            self.binary_task = {}
            self.graded_task = {Target(word): score for word, score in scores.items()}


    def evaluate_cd(self, predictions):
        """
            Evaluates binary change detection by comparing the predictions to the change scores in self.binary_task.

            Args:
                predictions (Union[List[Int], Dict[Str: Int]]): either a list of predictions (0 or 1) in the same order as the keys of self.stats_groupings or a dictionary {target_word: prediction}.

            Returns:
                (numpy.float64) An accuracy score: the percentage of correct predictions.
        """
        if self.binary_task == {}:
            logging.error('The dataset does not contain binary change scores; nothing to evaluate on.')
            return

        if type(predictions) == list:
            return accuracy_score(list(self.binary_task.values()), predictions)
        
        elif type(predictions) == dict:
            sorted_binary_scores = [i[1] for i in sorted(self.binary_task.items(), key = lambda i : getattr(i[0], 'lemma', str(i[0])))]
            sorted_predictions = [i[1] for i in sorted(predictions.items(), key = lambda i : i[0])]
            return accuracy_score(sorted_binary_scores, sorted_predictions)
    
    def evaluate_gcd(self, predictions):
        """
            Evaluates graded change detection by comparing the predictions to the change scores in self.graded_task.

            Args:
                predictions (Union[List[Int], Dict[Str: Int]]): either a list of predictions (0 or 1) in the same order as the keys of self.stats_groupings or a dictionary {target_word: prediction}.

            Returns:
                (scipy.stats._stats_py.SignificanceResult[numpy.float64, numpy.float64]) The Spearman correlation (rho, p) between the predictions and the gold labels.
        """
        if self.graded_task == {}:
            logging.error('The dataset does not contain graded change scores; nothing to evaluate on.')
            return
        
        if type(predictions) == list:
            return spearmanr(list(self.graded_task.values()), predictions)
        
        elif type(predictions) == dict:
            sorted_graded_scores = [i[1] for i in sorted(self.graded_task.items(), key = lambda i : getattr(i[0], 'lemma', str(i[0])))]
            sorted_predictions = [i[1] for i in sorted(predictions.items(), key = lambda i : i[0])]
            return spearmanr(sorted_graded_scores, sorted_predictions)


class SemEval2020Task1(SemanticChangeEvaluationDataset):

    def __init__(self, language, subset:int=None, config='opt'):
        lc = LanguageChange()
        self.language = language
        if self.language == 'NO' or self.language == 'RU':
            self.subset = subset
        if self.language == 'NO':
            self.dataset = 'NorDiaChange'
        elif self.language == 'RU':
            self.dataset = 'RuShiftEval'
        else:
            self.dataset = 'SemEval 2020 Task 1'
        home_path = lc.get_resource('benchmarks', self.dataset, self.language, 'no-version')
        semeval_folder = os.listdir(home_path)[0]
        self.home_path = os.path.join(home_path,semeval_folder)
        self.config = config
        self.load()

    def load(self):
        if self.dataset == "SemEval 2020 Task 1":
            self.corpus1_lemma = LinebyLineCorpus(os.path.join(self.home_path, 'corpus1', 'lemma'), name='corpus1_lemma', language=self.language, time=NumericalTime(1), is_lemmatized=True)
            self.corpus2_lemma = LinebyLineCorpus(os.path.join(self.home_path, 'corpus2', 'lemma'), name='corpus2_lemma', language=self.language, time=NumericalTime(2), is_lemmatized=True)
            self.corpus1_token = LinebyLineCorpus(os.path.join(self.home_path, 'corpus1', 'token'), name='corpus1_token', language=self.language, time=NumericalTime(1), is_tokenized=True)
            self.corpus2_token = LinebyLineCorpus(os.path.join(self.home_path, 'corpus2', 'token'), name='corpus2_token', language=self.language, time=NumericalTime(2), is_tokenized=True)

        self.binary_task = {}
        self.graded_task = {}
        self.target_words = set()

        if self.language == 'NO':
            matches = list(Path(os.path.join(self.home_path, f'subset{self.subset}','stats',self.config)).glob('stats_groupings.[ct]sv'))
            if matches == []:
                logging.error(f"Path does not exist: {os.path.join(self.home_path, f'subset{self.subset}','stats',self.config)}/stats_groupings.[ct]sv")
                raise FileNotFoundError
            with open(matches[0]) as f:
                for line in islice(f, 1, None):
                    line = line.strip('\n').split('\t')
                    word, binary, graded = line[0], line[11], line[14]
                    self.target_words.add(word)
                    word = Target(word)
                    self.binary_task[word] = int(binary)
                    self.graded_task[word] = float(graded)

        elif self.language == 'RU':
            # For the Russian dataset there are no binary change scores.
            matches = list(Path(self.home_path).glob('annotated_all.[ct]sv'))
            if matches == []:
                logging.error(f"Path does not exist: {self.home_path}/annotated_all.[ct]sv")
                raise FileNotFoundError
            with open(matches[0]) as f:
                for line in f:
                    line = line.strip('\n').split('\t')
                    word = line[0]
                    self.target_words.add(word)
                    word = Target(word)
                    # RuShiftEval uses the COMPARE metric, which measures relatedness between timeperiods, i.e. inverted semantic change.
                    self.graded_task[word] = -float(line[self.subset])

        else:
            with open(os.path.join(self.home_path, 'truth', 'binary.txt')) as f:
                for line in f:
                    word, label = line.split()
                    self.target_words.add(word)
                    word = Target(word)
                    self.binary_task[word] = int(label)

            with open(os.path.join(self.home_path, 'truth', 'graded.txt')) as f:
                for line in f:
                    word, score = line.split()
                    self.target_words.add(word)
                    word = Target(word)
                    self.graded_task[word] = float(score)

    def get_word_usages(self, word, group='all'):
        group = str(group)
        usages = TargetUsageList()
        if self.dataset == 'NorDiaChange':
            matches = list(Path(os.path.join(self.home_path,f'subset{self.subset}','data',word)).glob('uses.[ct]sv'))
            if matches == []:
                logging.error(f"Path does not exist: {os.path.join(self.home_path,f'subset{self.subset}','data',word)}/uses.(c|t)sv")
                raise FileNotFoundError
            with open(matches[0]) as f:
                keys = []
                for j,line in enumerate(f):
                    line = line.replace('\n','').split('\t')
                    if j > 0:
                        values = line
                        D = {keys[j]:values[j] for j in range(len(values))}
                        if group == 'all' or D['grouping'] == group:
                            D['text'] = D['context']
                            D['target'] = Target(D['lemma'])
                            D['target'].set_lemma(D['lemma'])
                            D['target'].set_pos(D['pos'])
                            D['offsets'] = [int(i) for i in D['indexes_target_token'].split(':')]
                            D['time'] = LiteralTime(D['date'])
                            usages.append(DWUGUsage(**D))
                    else:
                        keys = line
        elif self.dataset == 'RuShiftEval':
            matches = list(Path(os.path.join(self.home_path,'durel',f'rushifteval{self.subset}','data',word)).glob('uses.[ct]sv'))
            if matches == []:
                logging.error(f"Path does not exist: {os.path.join(self.home_path,'durel',f'rushifteval{self.subset}','data',word)}/uses.(c|t)sv")
                raise FileNotFoundError
            with open(matches[0]) as f:
                keys = []
                for j,line in enumerate(f):
                    line = line.replace('\n','').split('\t')
                    if j > 0:
                        values = line
                        D = {keys[j]:values[j] for j in range(len(values))}
                        D['text'] = D['context']
                        D['target'] = Target(D['lemma'])
                        D['target'].set_lemma(D['lemma'])
                        D['target'].set_pos(D['pos'])
                        D['offsets'] = [int(i) for i in D['indexes_target_token'].split(':')]
                        D['time'] = LiteralTime(D['date'])
                        usages.append(DWUGUsage(**D))
                    else:
                        keys = line
        return usages


class DWUG(SemanticChangeEvaluationDataset):

    def __init__(self, path=None, dataset=None, language=None, version=None, subset=None, config='opt'):
        lc = LanguageChange()
        self.dataset = dataset
        self.language = language
        self.version = version
        self.subset = subset
        if path == None and not dataset == None and not language == None and not version == None:
            home_path = lc.get_resource('benchmarks', self.dataset, self.language, version)
            if self.dataset == "NorDiaChange":
                dwug_folder = os.path.join(os.listdir(home_path)[0], f'subset{self.subset}')
            else:
                dwug_folder = os.listdir(home_path)[0]  
            self.home_path = os.path.join(home_path,dwug_folder)
        else:
            if not path == None and os.path.exists(path):
                self.home_path = path
            else:
                raise Exception('The path is None or does not exist.')   
        self.config = config    
        self.load()

    def load(self):
        self.target_words = os.listdir(os.path.join(self.home_path,'data'))
        self.stats_groupings = {}
        self.stats = {}

        stats_path = None
        if not self.config == None:
            stats_path = os.path.join(self.home_path,'stats',self.config)
        elif os.path.exists(os.path.join(self.home_path,'stats','opt')):
            stats_path = os.path.join(self.home_path,'stats','opt')
        elif os.path.exists(os.path.join(self.home_path,'stats')):
            stats_path = os.path.join(self.home_path,'stats')

        matches = list(Path(stats_path).glob('stats_groupings.[ct]sv'))
        if matches != []:
            with open(matches[0]) as f:
                keys = []
                for j,line in enumerate(f):
                    line = line.replace('\n','').split('\t')
                    if j > 0:
                        values = line
                        D = {keys[j]:values[j] for j in range(1,len(values))}
                        self.stats_groupings[values[0]] = D
                    else:
                        keys = line

        if stats_path is not None:
            matches = list(Path(stats_path).glob('stats.[ct]sv'))
            if matches != []:
                with open(matches[0]) as f:
                    keys = []
                    for j,line in enumerate(f):
                        line = line.replace('\n','').split('\t')
                        if j > 0:
                            values = line
                            D = {keys[j]:values[j] for j in range(1,len(values))}
                            self.stats[values[0]] = D
                        else:
                            keys = line
            else:
                logging.info("Could not find a path to stats.[c|t]sv. Did you enter the right config value?")

        self.binary_task = {}
        self.graded_task = {}
        self.binary_gain_task = {}
        self.binary_loss_task = {}

        for lemma in self.stats_groupings:

            word = Target(lemma)
            word.set_lemma(lemma)
            if 'change_binary' in self.stats_groupings[lemma].keys():
                self.binary_task[word] = int(self.stats_groupings[lemma]['change_binary'])
            if 'change_graded' in self.stats_groupings[lemma].keys():
                self.graded_task[word] = float(self.stats_groupings[lemma]['change_graded'])
            if 'change_binary_gain' in self.stats_groupings[lemma].keys():
                self.binary_gain_task[word] = int(self.stats_groupings[lemma]['change_binary_gain'])
            if 'change_binary_loss' in self.stats_groupings[lemma].keys():
                self.binary_loss_task[word] = int(self.stats_groupings[lemma]['change_binary_loss'])


    def get_usage_graph(self, word):
        with open(os.path.join(self.home_path,'graphs','opt',word),'rb') as f:
            return pickle.load(f)

    def show_usage_graph(self, word, config=None):
        def run_from_ipython():
            try:
                __IPYTHON__
                return True
            except NameError:
                return False

        def search_plot_path(path):
            if 'weight' in os.listdir(path):
                return path
            else:
                return search_plot_path(os.path.join(path,os.listdir(path)[0]))

        plot_path = None

        if config == None:
            path = search_plot_path(os.path.join(self.home_path,'plots'))
            plot_path = os.path.join(path,'weight','full')   
        else:
            plot_path = os.path.join(self.home_path,'plots',config,'weight','full') 


        if not run_from_ipython():
            webbrowser.open(os.path.join(plot_path,f'{word}.html'))
        else:
            from IPython.display import display, HTML
            with open(os.path.join(plot_path,f'{word}.html')) as f:
                html = f.read()
                display(HTML(html))

    def get_word_usages(self, word, group='all'):
        group = str(group)
        usages = TargetUsageList()
        matches = list(Path(os.path.join(self.home_path,'data',word)).glob('uses.[ct]sv'))

        if matches == []:
            logging.error(f"Did not find {os.path.join(self.home_path,'data',word)}/uses.(c|t)sv.")
            raise FileNotFoundError
        
        with open(matches[0]) as f:
            keys = []
            for j,line in enumerate(f):
                line = line.replace('\n','').split('\t')
                if j > 0:
                    values = line
                    D = {keys[j]:values[j] for j in range(len(values))}
                    if group == 'all' or D['grouping'] == group:
                        D['text'] = D['context']
                        D['target'] = Target(D['lemma'])
                        D['target'].set_lemma(D['lemma'])
                        D['target'].set_pos(D['pos'])
                        D['offsets'] = [int(i) for i in D['indexes_target_token'].split(':')]
                        D['time'] = LiteralTime(D['date'])
                        usages.append(DWUGUsage(**D))
                else:
                    keys = line
        return usages

    def annotate_word(self,
            word,
            model,
            metric: str | Callable,
            n_judgments: int | str="all",
            prompt_template = """Please tell me how similar the meaning of the word \'{target}\' 
            is in the following example sentences: \n1. {usage_1}\n2. {usage_2}"""):
        """
            Compares all usages of the target word in question and uses a model to compute judgments of their pairwise similarities, and saves the judgments to data/word/judgments.csv.
            Args:
                word (str): the target word to annotate.
                model (Union[ContextualizedModel, DefinitionGenerator, PromptModel]): the model to use to annotate the usages.
                metric (str or Callable): if a ContextualizedModel or DefinitionGenerator is used, the metric to use to compute similarity between two 
                    vectors. Supported string values are 'cosine', 'durel' and 'binary'. Alternatively, a function taking two vectors as input and 
                    returning a similarity score can be passed. If a PromptModel is used, this argument is ignored.
                prompt_template (str): if a PromptModel is used, the template to use for the user message in the prompt. 
                    The template must contain the placeholders '{target}', '{usage_1}' and '{usage_2}'.
        """
        def generate_index_pairs(total, n = "all"):
            all_index_pairs = [(i, j) for i in range(total) for j in range(i+1,total)]
            if n != "all":
                assert n > 0, "Cannot randomly choose a negative amount of samples"
                rng = np.random.default_rng()
                return rng.choice(all_index_pairs, n, replace=False)
            return all_index_pairs

        usages = self.get_word_usages(word)
        similarity_scores = {}

        if isinstance(model, ContextualizedModel) or isinstance(model, DefinitionGenerator):
            if isinstance(model, ContextualizedModel):
                embeddings = model.encode(usages)
            elif isinstance(model, DefinitionGenerator):
                embeddings = model.generate_definitions(usages, encode_definitions = 'vectors')
            
            if isinstance(metric, str):
                if metric.lower() == 'cosine':
                    similarity_func = lambda e1, e2 : np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                elif metric.lower() == 'durel':
                    similarity_func = lambda e1, e2 : math.ceil(4 * (np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))))
                elif metric.lower() == 'binary':
                    similarity_func = lambda e1, e2 : int(np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2)) > 0.5)
                else:
                    logging.error(f"Metric '{metric}' not recognized. Supported metrics are 'cosine', 'durel' and 'binary'.")
                    raise ValueError
            elif callable(metric):
                signature = inspect.signature(similarity_func)
                n_req_args = sum([int(p.default == p.empty) for p in signature.parameters.values()])
                if n_req_args != 2:
                    logging.error(f"'label_func' must take 2 arguments but takes {n_req_args}.")
                    return None
                similarity_func = metric

            index_pairs = generate_index_pairs(len(embeddings), n = n_judgments)
                
            for i, j in index_pairs:
                id1, id2 = usages[i].identifier, usages[j].identifier
                similarity_scores[frozenset([id1, id2])] = similarity_func(embeddings[i], embeddings[j])

        elif isinstance(model, PromptModel):
            index_pairs = generate_index_pairs(len(usages), n = n_judgments)

            for i, j in index_pairs:
                u1, u2 = usages[i], usages[j]
                id1, id2 = u1.identifier, u2.identifier
                similarity_scores[frozenset([id1, id2])] = model.get_response([u1, u2], user_prompt_template = prompt_template)

        judgments_f = os.path.join(self.home_path,'data',word,'judgments.csv')
        with open(judgments_f, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['identifier1', 'identifier2', 'annotator', 'judgment', 'lemma'])
            for ids, similarity in similarity_scores.items():
                ids = list(ids)
                id1, id2 = ids[0], ids[1]
                writer.writerow([id1, id2, type(model).__name__, similarity, word])
            logging.info(f"Usages for {word} annotated by {type(model).__name__} and written to {judgments_f}.")

    def annotate_all(self, model, metric : str | Callable = None, n_judgments : int | str = "all",
            prompt_template = 'Please tell me how similar the meaning of the word \'{target}\' is in the following example sentences: \n1. {usage_1}\n2. {usage_2}',):
        """
            Annotates all target words in the dataset using the given model and metric (if applicable), 
            and saves the judgments to data/word/judgments.csv. Parameters as in self.annotate_word.
        """
        for word in self.target_words:
            self.annotate_word(word, model, metric, n_judgments, prompt_template)

    def cluster(self,
                word,
                s=20,
                max_attempts=2000,
                max_iters=50000,
                initial = [],
                split_flag = True,
                plot = True,
                outfile = None,
                plot_id_labels = True):

        def load_graph(judgments_f):
            G = nx.Graph()
            df = pd.read_csv(judgments_f, sep="\t")
            
            for _, row in df.iterrows():
                id1 = row["identifier1"]
                id2 = row["identifier2"]
                w = float(row["judgment"])
                G.add_edge(id1, id2, weight=w)
            return G

        def transform_edge_weights(G, transformation = lambda x : x):
            for (i,j) in G.edges():
                G[i][j]['weight'] = transformation(G[i][j]['weight'])

        # Load the judgments as a graph
        G = load_graph(os.path.join(self.home_path,'data',word,'judgments.csv'))
        transform_edge_weights(G, transformation = lambda w : w - 2.5)

        # Perform correlation clustering on the similarity graph
        clustering = Clustering(CorrelationClustering(s=s, max_attempts=max_attempts,
        max_iters=max_iters, initial=initial, split_flag=split_flag))
        clustering_results = clustering.get_cluster_results(G)
        cluster_labels = clustering_results.labels

        # Write the clustering labels to the clusters file of the word
        clusters_path = f"{self.home_path}/clusters/{self.config}/{word}.csv"
        with open(clusters_path, "w") as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(["identifier", "cluster"])
            for identifier, label in zip(G.nodes, cluster_labels):
                w.writerow([identifier, label])
        logging.info(f"Wrote cluster labels to {clusters_path}")

        if plot:
            self.plot_clustering(G, cluster_labels, outfile=outfile, plot_id_labels=plot_id_labels)

        return G, cluster_labels

    def plot_clustering(self, G, classes, outfile : str = None, plot_id_labels = True):
        pos = nx.spring_layout(G, seed=42)

        weights = [G[u][v]["weight"] for u, v in G.edges()]
        edge_widths = [0.5 + 0.5 * abs(w) for w in weights]

        plt.figure(figsize=(10, 8))
        plt.axis("off")

        # If classes are supplied, plot each node in a color corresponding to the class
        if classes is not None:
            cmap = plt.cm.turbo
            norm = matplotlib.colors.Normalize(vmin=0, vmax=len(set(classes)))
            discrete_cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).get_cmap()

            nx.draw_networkx_nodes(G, pos, node_size=60, node_color=classes, cmap=discrete_cmap)
        else:
            nx.draw_networkx_nodes(G, pos, node_color="blue", node_size=60)

        nx.draw_networkx_edges(G, pos, edge_color="grey", width=edge_widths)

        if plot_id_labels:
            nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

        plt.tight_layout()
        if outfile is not None:
            plt.savefig(outfile, dpi=300)
            logging.info(f"Plot saved to {outfile}.")
        else:
            plt.show()
        plt.close()

    def annotate_and_cluster(self,
            word,
            annotator,
            metric,
            n_judgments : int | str = "all",
            prompt_template = 'Please tell me how similar the meaning of the word \'{target}\' is in the following example sentences: \n1. {usage_1}\n2. {usage_2}',
            s = 20,
            max_attempts = 2000,
            max_iters = 50000,
            initial = [],
            split_flag = True, 
            plot = True, 
            outfile = None, 
            plot_id_labels = True):
        self.annotate_word(word, annotator, metric, n_judgments=n_judgments, prompt_template=prompt_template)
        self.cluster(word, s=s, max_attempts=max_attempts, max_iters=max_iters, initial=initial, split_flag=split_flag, plot=plot, outfile=outfile,
        plot_id_labels=plot_id_labels)

    def annotate_and_cluster_all(self, annotator, metric, **kwargs):
        for word in self.target_words:
            self.annotate_and_cluster(word, annotator, metric, **kwargs)

    def _get_outliers(self, word):
        """
            Finds all usages of a given word which have been marked as outliers in the clustering step (cluster label = -1).
            Args:
                word (str): the target word to find outliers for.
            Returns:
                (set) a set of identifiers of usages which are outliers.
        """
        outliers = set()

        try:
            if self.dataset == "DWUG Sense":
                clusters_path = os.path.join(self.home_path,'labels',word,self.config)
                matches = list(Path(clusters_path).glob('labels_senses.[ct]sv'))
            else:
                clusters_path = os.path.join(self.home_path,'clusters',self.config)
                matches = list(Path(clusters_path).glob(f'{word}.[ct]sv'))
            with open(matches[0]) as f:
                for line in islice(f, 1, None):
                    line = line.replace('\n','').split('\t')
                    id, label = line
                    try:
                        if int(label) == -1:
                            outliers.add(id)
                    except ValueError:
                        continue
        except Exception as e:
            logging.error(f"Could not remove outlier usages of '{word}' due to the following error: {e}")
            raise e
        
        return outliers

    def get_word_annotations(self, word, only_between_groups = False, remove_outliers = False, exclude_non_judgments = False, transform_labels : Callable | str = None, return_list = False):
        """
            Finds the judgments for a given word in the DWUG.
            Args:
                only_between_groups (bool) : if true, select only examples where the two usages belong to different groupings.
                remove_outliers (bool) : if true, remove all examples which have been not been assigned to a cluster (cluster label = -1).
                exclude_non_judgments (bool): if true, remove all pairs of usages for which there is no judgment (label = 0).
                transform_labels (Callable): a function which takes a list of labels and returns a label. By default, all labels are kept. As a string, only 'mean' is supported.
                return_list (bool): if true, return the judgments as a list.
            Returns:
                (Dict[frozenset, Dict] or List[Dict]) a dictionary {frozenset([id1, id2]) : {'word': word, 'id1': id1, 'text1': text1, 'start1': start1, 'end1': end1, 'id2': id2, 'text2': text2, 'start2': start2, 'end2': end2, 'label': label}} or a list of such dictionaries if return_list is true.
        """
        
        judgments = {}

        if remove_outliers:
            excluded_instances = self._get_outliers(word)
        else:
            excluded_instances = set()

        usages_by_key = {}
        matches = list(Path(os.path.join(self.home_path,'data',word)).glob('uses.[ct]sv'))
        try:
            with open(matches[0]) as f:
                for line in islice(f, 0, 1):
                    columns = line.replace('\n','').split('\t')
                    column_ids = {c : i for i,c in enumerate(columns)}

                if 'context_tokenized' in column_ids and 'indexes_target_token_tokenized' in column_ids:
                    for line in f:
                        line = line.replace('\n','').split('\t')
                        id = line[column_ids['identifier']]
                        if not id in excluded_instances:
                            lemma = line[column_ids['lemma']]
                            grouping = line[column_ids['grouping']]
                            context = line[column_ids['context_tokenized']]
                            word_index = int(line[column_ids['indexes_target_token_tokenized']])
                            start, end = self.word_index_to_char_indices(context, word_index, split_text=True)
                            usages_by_key[id] = {'word': lemma, 'text':context, 'start':start, 'end':end, 'grouping':grouping}
                else:
                    for line in f:
                        line = line.replace('\n','').split('\t')
                        id = line[column_ids['identifier']]
                        if not id in excluded_instances:
                            lemma = line[column_ids['lemma']]
                            grouping = line[column_ids['grouping']]
                            context = line[column_ids['context']]
                            start, end = line[column_ids['indexes_target_token']].split(":")
                            start, end = int(start), int(end)
                            usages_by_key[id] = {'word': lemma, 'text':context, 'start':start, 'end':end, 'grouping':grouping}
        except Exception as e:
            logging.error(f"Could not load usage data for '{word}' due to the following error: {e}")
            raise e

        temp_labels = {}
        try:
            if self.dataset == "DWUG Sense":
                matches = list(Path(os.path.join(self.home_path,"labels",word,self.config)).glob('labels_proximity.[ct]sv'))
                with open(matches[0]) as f:
                    for line in islice(f, 0, 1):
                        columns = line.replace('\n','').split('\t')
                        column_ids = {c : i for i,c in enumerate(columns)}
                    for line in f:
                        line = line.replace('\n','').split('\t')
                        id1, id2 = line[column_ids['identifier1']], line[column_ids['identifier2']]
                        if id1 != id2:
                            label = float(line[column_ids['label']])
                            if not id1 in excluded_instances and not id2 in excluded_instances:
                                if not frozenset([id1,id2]) in temp_labels:
                                    temp_labels[frozenset([id1,id2])] = []
                                temp_labels[frozenset([id1,id2])].append(label)
            else:
                matches = list(Path(os.path.join(self.home_path,"data",word)).glob('judgments.[ct]sv'))
                with open(matches[0]) as f:
                    for line in islice(f, 0, 1):
                        columns = line.replace('\n','').split('\t')
                        column_ids = {c : i for i,c in enumerate(columns)}
                    for line in f:
                        line = line.replace('\n','').split('\t')
                        id1, id2 = line[column_ids['identifier1']], line[column_ids['identifier2']]
                        if id1 != id2:
                            label = float(line[column_ids['judgment']])
                            if not (label == 0 and exclude_non_judgments) and not id1 in excluded_instances and not id2 in excluded_instances:
                                if not frozenset([id1,id2]) in temp_labels:
                                    temp_labels[frozenset([id1,id2])] = []
                                temp_labels[frozenset([id1,id2])].append(label)
        except Exception as e:
            logging.error(f"Could not load judgment data for '{word}' due to the following error: {e}")
            raise e

        try:
            for key in temp_labels:
                ordered_ids = list(key)
                id1, id2 = ordered_ids[0], ordered_ids[1]
                usage1, usage2 = usages_by_key[id1], usages_by_key[id2]
                word = usage1['word']
                assert word == usage2['word']
                if only_between_groups and usage1['grouping'] == usage2['grouping']:
                    continue
                if transform_labels is None:
                    label = temp_labels[key]
                elif callable(transform_labels):
                    try:
                        label = transform_labels(temp_labels[key])
                    except:
                        logging.error(f'{transform_labels} could not be used as a function to transform labels.')
                        raise ValueError
                elif type(transform_labels) == str:
                    try:
                        transform_funcs = {
                            'mean' : lambda l : np.mean(l)
                        }
                        label = transform_funcs[transform_labels](temp_labels[key])
                    except KeyError:
                        logging.error(f"{transform_labels} does not denote a function used to transform the list of labels into one label. Currently only 'mean' is supported.")
                        raise KeyError
                    
                judgments[frozenset([id1, id2])] = {'word': word, 
                            'id1': id1, 'text1': usage1['text'], 'start1': usage1['start'], 'end1': usage1['end'],
                            'id2': id2, 'text2': usage2['text'], 'start2': usage2['start'], 'end2': usage2['end'],
                            'label': label}
        except Exception as e:
            logging.error(f"Could not combine usage and judgment data for '{word}' due to the following error: {e}")
            raise e
        
        if return_list:
            return judgments.values()
        return judgments

    def get_stats(self):
        return self.stats

    def get_stats_groupings(self):
        return self.stats_groupings
    
    def cast_to_WiC(self, only_between_groups = False, remove_outliers = True, exclude_non_judgments = True, transform_labels : Callable | str = 'mean'):
        """
            Casts the DWUG to a Word in Context (WiC) dataset.

            Args:
                only_between_groups (bool) : if true, select only examples where the two usages belong to different groupings.
                remove_outliers (bool) : if true, remove all examples which have been not been assigned to a cluster (cluster label = -1).
                exclude_non_judgments (bool): if true, remove all pairs of usages for which there is no judgment (label = 0).
                transform_labels (Callable|str): a function or a string denoting a function (see self.get_word_annotation) which takes a list of labels and returns a label, by default the mean of the labels.
        """
        data = []
        for word in self.target_words:
            judgments = self.get_word_annotations(word, only_between_groups=only_between_groups, remove_outliers=remove_outliers, exclude_non_judgments=exclude_non_judgments, transform_labels=transform_labels)
            data.extend(judgments.values())

        wic = WiC(wic_data=data, dataset=f'{self.dataset} WiC' if self.dataset is not None else None, language=self.language, version=self.version, subset=self.subset)
        return wic
    
    def get_usages_and_senses(self, remove_outliers = True):
        data = []
        for word in self.target_words:
            usages_by_id = {}

            
            if self.dataset == "DWUG Sense": #This is not good
                senses_path = os.path.join(self.home_path,'labels',word,self.config)
                matches = list(Path(senses_path).glob('labels_senses.[ct]sv'))
            else:
                senses_path = os.path.join(self.home_path,f'clusters/{self.config}')
                matches = list(Path(senses_path).glob(f'{word}.[ct]sv'))

            try:
                with open(matches[0]) as f:
                    for line in islice(f, 0, 1):
                        columns = line.replace('\n','').split('\t')
                        column_ids = {c : i for i,c in enumerate(columns)}
                    for line in f:
                        line = line.replace('\n','').split('\t')
                        id = line[column_ids['identifier']]
                        label = line[column_ids['label'] if self.dataset == "DWUG Sense" else column_ids['cluster']]
                        try:
                            if not (remove_outliers and int(label) == -1):
                                usages_by_id[id] = {'id': id, 'label': label}
                        except ValueError:
                            usages_by_id[id] = {'id': id, 'label': label}
            except Exception as e:
                logging.error(f"Could not load sense labels for '{word}' due to the following error: {e}")
                raise e

            try:
                matches = list(Path(os.path.join(self.home_path,'data',word)).glob('uses.[ct]sv'))
                with open(matches[0]) as f:
                    for line in islice(f, 0, 1):
                        columns = line.replace('\n','').split('\t')
                        column_ids = {c : i for i,c in enumerate(columns)}
                    
                    if 'context_tokenized' in column_ids and 'indexes_target_token_tokenized' in column_ids:
                        for line in f:
                            line = line.replace('\n','').split('\t')
                            id = line[column_ids['identifier']]
                            if id in usages_by_id:
                                lemma = line[column_ids['lemma']]
                                context = line[column_ids['context_tokenized']]
                                word_index = int(line[column_ids['indexes_target_token_tokenized']])
                                start, end = self.word_index_to_char_indices(context, word_index, split_text=True)
                                usages_by_id[id].update({'word': lemma, 'text':context, 'start':start, 'end':end, 'label': lemma + ":" + usages_by_id[id]['label']})
                    else:
                        for line in f:
                            line = line.replace('\n','').split('\t')
                            id = line[column_ids['identifier']]
                            if id in usages_by_id:
                                lemma = line[column_ids['lemma']]
                                context = line[column_ids['context']]
                                start, end = line[column_ids['indexes_target_token']].split(":")
                                start, end = int(start), int(end)
                                usages_by_id[id].update({'word': lemma, 'text':context, 'start':start, 'end':end, 'label': lemma + ":" + usages_by_id[id]['label']})
            except Exception as e:
                logging.error(f"Could not load usages for '{word}' due to the following error: {e}")
                raise e
            
            for id, ex in usages_by_id.items():
                for k in {'text','start','end','word','label'}:
                    if not k in ex:
                        logging.error(f"A value for {k} in missing in the example of id {id}. Make sure that {senses_path} and {os.path.join(self.home_path,'data',word,'uses.(c|t)sv')} contain the same examples.")
                        raise KeyError
            
            data.extend(list(usages_by_id.values()))
        return data
        
    def cast_to_WSD(self, remove_outliers = True):
        data = self.get_usages_and_senses(remove_outliers)
        wsd = WSD(wsd_data=data, dataset=f'{self.dataset} WSD' if self.dataset is not None else None, language=self.language, version=self.version, subset=self.subset)
        return wsd

    def cast_to_WSI(self, remove_outliers = True):
        data = self.get_usages_and_senses(remove_outliers)
        wsi = WSI(wsi_data=data, dataset=f'{self.dataset} WSI' if self.dataset is not None else None, language=self.language, version=self.version, subset=self.subset)
        return wsi
    
    def cluster_evaluation(self, predictions, metrics = {'ari', 'purity'}, remove_outliers = True):
        wsi = self.cast_to_WSI(remove_outliers)
        return wsi.evaluate(predictions, metrics)
    
    def evaluate_cd(self, predictions):
        """
            Evaluates binary change detection by comparing the predictions to the change scores in self.binary_task.

            Args:
                predictions (Union[List[Int], Dict[Str: Int]]): either a list of predictions (0 or 1) in the same order as the keys of self.stats_groupings or a dictionary {target_word: prediction}.

            Returns:
                (numpy.float64) An accuracy score: the percentage of correct predictions.
        """
        if self.binary_task == {}:
            logging.error('DWUG does not contain binary change scores; nothing to evaluate on.')
            return

        if type(predictions) == list:
            return accuracy_score(list(self.binary_task.values()), predictions)
        
        elif type(predictions) == dict:
            sorted_binary_scores = [i[1] for i in sorted(self.binary_task.items(), key = lambda i : getattr(i[0], 'lemma', str(i[0])))]
            sorted_predictions = [i[1] for i in sorted(predictions.items(), key = lambda i : i[0])]
            return accuracy_score(sorted_binary_scores, sorted_predictions)
    
    def evaluate_gcd(self, predictions):
        """
            Evaluates graded change detection by comparing the predictions to the change scores in self.graded_task.

            Args:
                predictions (Union[List[Int], Dict[Str: Int]]): either a list of predictions (0 or 1) in the same order as the keys of self.stats_groupings or a dictionary {target_word: prediction}.

            Returns:
                (scipy.stats._stats_py.SignificanceResult[numpy.float64, numpy.float64]) The Spearman correlation (rho, p) between the predictions and the gold labels.
        """
        if self.graded_task == {}:
            logging.error('DWUG does not contain graded change scores; nothing to evaluate on.')
            return
        
        if type(predictions) == list:
            return spearmanr(list(self.graded_task.values()), predictions)
        
        elif type(predictions) == dict:
            sorted_graded_scores = [i[1] for i in sorted(self.graded_task.items(), key = lambda i : getattr(i[0], 'lemma', str(i[0])))]
            sorted_predictions = [i[1] for i in sorted(predictions.items(), key = lambda i : i[0])]
            return spearmanr(sorted_graded_scores, sorted_predictions)
        

class WiC(Benchmark):
    """
        Dataset handling for the Word-in-Context (WiC) task.
        Parameters:
            path (str) : a path to the dataset, if it is not stored by the resource hub.
            dataset (str|list|dict) : the dataset to be loaded. One of ['WiC', 'XL-WiC', 'TempoWiC', 'MCL-WiC', 'AM2iCo'] if using a dataset in the language change resource hub, or a list or a dict if loading from a datastructure already describing a WiC dataset.
            version (str) : the version of the dataset if using a dataset from the resource hub.
            language (str) : the language code (e.g. AR), if loading a multi- or crosslingual dataset.
            linguality (str) : whether to use the crosslingual or multilingual dataset, in the case of MCL-WiC.
            name (str) : the name of the dataset (in case no values for dataset, language and version are specified).
    """
    def __init__(self, 
                 path : str = None, 
                 wic_data : dict | list = None,
                 dataset : str = None, 
                 version : str = None, 
                 language : str = None, 
                 linguality : str = None,
                 subset : str = None,
                 name : str = None):
        self.data = {}
        self.dataset = dataset
        self.version = version
        self.language = language
        self.linguality = linguality
        self.subset = subset
        self.target_words = set()
        if name is not None:
            self.name = name

        # Get the dataset from the resource hub, or load a locally stored dataset from files
        if wic_data == None and dataset != None and version != None and (dataset == 'WiC' or dataset == 'TempoWiC' or language != None):
            # Let the resource hub find the path
            if path == None:
                lc = LanguageChange()
                home_path = lc.get_resource('benchmarks', 'WiC', dataset, version)
                
                if dataset == 'XL-WiC' or dataset == 'TempoWiC' or dataset == 'MCL-WiC' or dataset == 'AM2iCo':
                    wic_folder = os.listdir(home_path)[0]
                    home_path = os.path.join(home_path, wic_folder)
                    if dataset == 'MCL-WiC':
                        if self.linguality not in {'multilingual','crosslingual'}:
                            logging.error("For MCL-WiC, linguality has to be set to either 'multilingual' or 'crosslingual'")
                            raise ValueError
                        self.language = f'EN-{self.language}' if self.linguality == 'crosslingual' else f'{self.language}-{self.language}'
                        if os.path.exists(os.path.join(home_path, "SemEval-2021_MCL-WiC_all-datasets.zip")):
                            with zipfile.ZipFile(os.path.join(home_path, "SemEval-2021_MCL-WiC_all-datasets.zip"), 'r') as f:
                                f.extractall(home_path)
                        if os.path.exists(os.path.join(home_path, "SemEval-2021_MCL-WiC_test-gold-data.zip")):
                            with zipfile.ZipFile(os.path.join(home_path, "SemEval-2021_MCL-WiC_test-gold-data.zip"), 'r') as f:
                                f.extractall(os.path.join(home_path, 'MCL-WiC'))
                        self.home_path = os.path.join(home_path, 'MCL-WiC')
                    else:
                        self.home_path = home_path
                else:
                    self.home_path = home_path
            # Pre-defined path
            else:
                self.home_path = path
            logging.info(f"WiC home path: {self.home_path}")
                
            self.load_from_resource_hub()

        # Loads from a dictionary or list
        elif wic_data != None:
            try:
                self.load_from_data(wic_data)
            except Exception as e:
                logging.error('Could not load from dataset.')
                raise e
    
    # Loads from a list or a dict containing a WiC dataset, with each example as in self.load_from_resource_hub()
    def load_from_data(self, data):
        if type(data) == list:
            self.data = {'all': data}
        elif type(data) == dict:
            self.data = data
        else:
            raise TypeError('Could not load the dataset as a WiC dataset.')

        # Add the lemma in each data example to the set of target words
        for dataset in self.data.values():
            for d in dataset:
                if 'word' in d.keys():
                    self.target_words.add(d['word'])

    # Finds the file paths of the data and labels for possible train, dev and test sets.
    def find_data_paths(self):
        train_paths = {'data':None, 'labels':None}
        dev_paths= {'data':None, 'labels':None}
        test_paths = {'data':None, 'labels':None}
        data_paths = {'train':train_paths, 'dev':dev_paths, 'test':test_paths}

        if self.dataset == 'WiC':
            for s in data_paths.keys():
                data_paths[s]['data'] = s + "/" + s + ".data.txt"
                data_paths[s]['labels'] = s + "/" + s + ".gold.txt"

        elif self.dataset == 'XL-WiC':
            language_paths = {'BG':'xlwic_wn/bulgarian_bg', 
                              'DA':'xlwic_wn/danish_da', 
                              'DE':'xlwic_wikt/german_de',
                              'EN':'wic_english',
                              'ET':'xlwic_wn/estonian_et',
                              'FA':'xlwic_wn/farsi_fa',
                              'FR':'xlwic_wikt/french_fr',
                              'HR':'xlwic_wn/croatian_hr',
                              'IT':'xlwic_wikt/italian_it',
                              'JA':'xlwic_wn/japanese_ja',
                              'KO':'xlwic_wn/korean_ko',
                              'NL':'xlwic_wn/dutch_nl',
                              'ZH':'xlwic_wn/chinese_zh'}
            try:
                language_path = language_paths[self.language]
            except KeyError:
                logging.error(f'Language {self.language} is not supported.')
                raise Exception

            # For English, train and dev sets are available, with both having labels.
            if self.language == 'EN':
                data_paths['train']['data'] = os.path.join(language_path, "train_en.txt")
                data_paths['dev']['data'] = os.path.join(language_path, "valid_en.txt")

            else:
                if os.path.exists(os.path.join(self.home_path, language_path, self.language.lower()+"_train.txt")):
                    data_paths['train']['data'] = os.path.join(language_path, self.language.lower()+"_train.txt")
                if os.path.exists(os.path.join(self.home_path, language_path, self.language.lower()+"_valid.txt")):
                    data_paths['dev']['data'] = os.path.join(language_path, self.language.lower()+"_valid.txt")
                if os.path.exists(os.path.join(self.home_path, language_path, self.language.lower()+"_test_data.txt")):
                    data_paths['test']['data'] = os.path.join(language_path, self.language.lower()+"_test_data.txt")
                if os.path.exists(os.path.join(self.home_path, language_path, self.language.lower()+"_test_gold.txt")):
                    data_paths['test']['labels'] = os.path.join(language_path, self.language.lower()+"_test_gold.txt")
    
        elif self.dataset == 'TempoWiC':
            data_paths['train']['data'] = "data/train.data.jl"
            data_paths['train']['labels'] = "data/train.labels.tsv"
            data_paths['dev']['data'] = "data/validation.data.jl"
            data_paths['dev']['labels'] = "data/validation.labels.tsv"
            data_paths['test']['data'] = "data/test-codalab-10k.data.jl"
            data_paths['test']['labels'] = "data/test.gold.tsv"

        elif self.dataset == 'MCL-WiC':
            # The multilingual task
            if self.linguality == 'multilingual':
                if self.language.lower() == 'en-en':
                    data_paths['train']['data'] = "training/training.en-en.data"
                    data_paths['train']['labels'] = "training/training.en-en.gold"
                data_paths['dev']['data'] = f"dev/multilingual/dev.{self.language.lower()}.data"
                data_paths['dev']['labels'] = f"dev/multilingual/dev.{self.language.lower()}.gold"
                data_paths['test']['data'] = f"test/multilingual/test.{self.language.lower()}.data"
                data_paths['test']['labels'] = f"test.{self.language.lower()}.gold"
            # The crosslingual task
            elif self.linguality == 'crosslingual':
                data_paths['test']['data'] = f"test/crosslingual/test.{self.language.lower()}.data"
                data_paths['test']['labels'] = f"test.{self.language.lower()}.gold"

        elif self.dataset == 'AM2iCo':

            language_path = "data/" + self.language.lower()
            if not os.path.exists(os.path.join(self.home_path, language_path)):
                logging.error(f'Path {os.path.join(self.home_path, language_path)} does not exist.')
                raise FileNotFoundError
            
            for s in data_paths.keys():
                if os.path.exists(os.path.join(self.home_path, language_path, f"{s}.tsv")):
                    data_paths[s]['data'] = os.path.join(language_path, f"{s}.tsv")

            # For German and Russian there is also dev_larger and test_larger.
            for s in ['dev','test']:
                if os.path.exists(os.path.join(self.home_path, language_path, f"{s}_larger.tsv")):
                    data_paths[s+'_larger'] = {'data': os.path.join(language_path, f"{s}_larger.tsv")}

        return data_paths

    def load_from_txt(self, 
                      filename, 
                      word_indexes : bool = False,
                      index_to_offsets = None,
                      field_map = {'word': 0, 'start1': 2, 'end1': 3, 'start2':4, 'end2':5, 'text1': 6 , 'text2': 7, 'label': 8},
                      skiplines = 0):
        
        if index_to_offsets is None:
            index_to_offsets = lambda text, index : self.word_index_to_char_indices(text, index, split_text=True)

        def get_line_data(line, field_map, word_indexes : bool = False):
            line_data = {}
            line_values = line.strip("\n").split("\t")

            for key in field_map:
                if field_map[key] < len(line_values):
                    line_data[key] = line_values[field_map[key]]

            if word_indexes:
                if not 'indexes' in line_data.keys():
                    raise KeyError
                i1, i2 = (int(i) for i in line_data['indexes'].split("-"))
                start1, end1 = index_to_offsets(line_data['text1'],i1)
                start2, end2 = index_to_offsets(line_data['text2'],i2)
                line_data = line_data | {"start1": start1, "end1": end1, "start2": start2, "end2": end2}
                del line_data['indexes']

            if 'label' in line_data.keys():
                line_data['label'] = self.format_label(line_data['label'])

            for key in ['start1','end1','start2','end2']:
                if key in line_data.keys():
                    line_data[key] = int(line_data[key])

            return line_data
        
        data = []
        with open(os.path.join(self.home_path, filename), 'r') as f:
            for line in islice(f, skiplines, None):
                data.append(get_line_data(line, field_map, word_indexes))
        return data

    def load_from_files(self, data_paths):
        data = {'train':[], 'dev':[], 'test':[]}

        # The original Word-in-Context dataset
        if self.dataset == 'WiC':
            for key in data_paths.keys():
                if data_paths[key]['data'] is not None:
                    data[key] = self.load_from_txt(data_paths[key]['data'], word_indexes=True, field_map={'word':0,'indexes':2, 'text1':3, 'text2':4})
                if data_paths[key]['labels'] is not None:
                    labels = self.load_from_txt(data_paths[key]['labels'], field_map={'label': 0})
                    data[key] = [d | labels[i] for i, d in enumerate(data[key])]

        # XL-WiC, containing WiC datasets for 12 more languages other than English.
        elif self.dataset == 'XL-WiC':
            # There is something unusual in the offsets of the XL-WiC dataset for Farsi.
            # The first index is a word index while the difference between the first and the second denotes
            # the length of the word.
            if self.language == 'FA':
                def get_start_end(text, start, end):
                    char_start = sum(len(s)+1 for s in text.split(" ")[:int(start)])
                    char_end = char_start + int(end) - int(start)
                    return char_start, char_end
                
            for key in ['train','dev']:
                if data_paths[key]['data'] is not None:
                    data[key] = self.load_from_txt(data_paths[key]['data'])
                    if self.language == 'FA':
                        for i, d in enumerate(data[key]):
                            data[key][i]['start1'], data[key][i]['end1'] = get_start_end(d['text1'], d['start1'], d['end1'])
                            data[key][i]['start2'], data[key][i]['end2'] = get_start_end(d['text2'], d['start2'], d['end2'])

            if data_paths['test']['data'] is not None:
                data['test'] = self.load_from_txt(data_paths['test']['data'])
                if self.language == 'FA':
                    for i, d in enumerate(data['test']):
                        data['test'][i]['start1'], data['test'][i]['end1'] = get_start_end(d['text1'], d['start1'], d['end1'])
                        data['test'][i]['start2'], data['test'][i]['end2'] = get_start_end(d['text2'], d['start2'], d['end2'])  
            
                if data_paths['test']['labels'] is not None:
                    labels = self.load_from_txt(data_paths['test']['labels'], field_map={'label': 0})
                    data['test'] = [d | labels[i] for i, d in enumerate(data['test'])]

        # TempoWiC, containing social media data annotated with dates.
        elif self.dataset == "TempoWiC":
            for key in data_paths.keys():
                data_dict = {}

                if data_paths[key]['labels'] is not None:
                    with open(os.path.join(self.home_path, data_paths[key]['labels'])) as f:
                        for i, line in enumerate(f):
                            id, label = line.strip('\n').split('\t')
                            data_dict[id] = {'label': self.format_label(label)}

                if data_paths[key]['data'] is not None:
                    with open(os.path.join(self.home_path, data_paths[key]['data'])) as f:
                        for line in f:
                            json_data = json.loads(line)
                            text1 = json_data['tweet1']['text']
                            text2 = json_data['tweet2']['text']
                            word = json_data['word']
                            start1, end1 = json_data['tweet1']['text_start'], json_data['tweet1']['text_end']
                            start2, end2 = json_data['tweet2']['text_start'], json_data['tweet2']['text_end']
                            if json_data['id'] in data_dict: 
                                data_dict[json_data['id']] = data_dict[json_data['id']] | {'id': json_data['id'], 'word': word, 'text1': text1, 'text2': text2, 'start1': start1, 'end1': end1,'start2': start2,'end2': end2}

                data[key] = list(data_dict.values())

        # For MCL-WiC there is an English training set, a multilingual development set, and crosslingual and multilingual test sets.
        elif self.dataset == "MCL-WiC":
            for key in data_paths.keys():
                data_dict = {}
                
                if data_paths[key]['data'] is not None:
                    with open(os.path.join(self.home_path, data_paths[key]['data'])) as f:
                        json_data = json.load(f)

                        for ex in json_data:
                            word = ex['lemma']
                            text1 = ex['sentence1']
                            text2 = ex['sentence2']

                            if self.linguality != 'crosslingual':
                                start1 = int(ex['start1'])
                                end1 = int(ex['end1'])
                                start2 = int(ex['start2'])
                                end2 = int(ex['end2'])
                            else:
                                # Some examples have multiple occurrences of the target word. We take the first one here.
                                start1, end1 = (int(offset) for offset in ex['ranges1'].split(",")[0].split("-"))
                                start2, end2 = (int(offset) for offset in ex['ranges2'].split(",")[0].split("-"))

                            data_dict[ex['id']] = {'id': ex['id'], 'word': word, 'text1': text1, 'text2': text2, 'start1': start1, 'end1': end1,'start2': start2,'end2': end2}

                if data_paths[key]['labels'] is not None:
                    with open(os.path.join(self.home_path, data_paths[key]['labels'])) as f:
                        labels = json.load(f)
                        for label in labels:
                            if label['id'] in data_dict:
                                data_dict[label['id']]['label'] = self.format_label(label['tag'])

                data[key] = list(data_dict.values())

        # AM2iCo contains crosslingual datasets for 14 languages paired with English.
        elif self.dataset == "AM2iCo":
            regex = re.compile(r'<word>(.*)</word>')
            tag_length = len('<word></word>')

            def extract_word_indexes(text, regex):
                match = regex.search(text)
                text = re.sub(regex, r"\1", text)
                start, end = match.span()
                end -= tag_length
                return text, start, end

            for key in data_paths.keys():
                if data_paths[key]['data'] is not None:
                    data[key] = self.load_from_txt(data_paths[key]['data'], field_map={'text1': 0, 'text2': 1, 'label': 2}, skiplines=1)
                    for i, d in enumerate(data[key]):
                        text1, start1, end1 = extract_word_indexes(d['text1'], regex)
                        text2, start2, end2 = extract_word_indexes(d['text2'], regex)
                        data[key][i] = d | {'text1':text1, 'start1':start1, 'end1':end1, 'text2':text2, 'start2':start2, 'end2':end2}

        self.load_from_data(data)
        
    def format_label(self, label):
        if label == 1 or label == 0:
            return label
        if label == '1' or label == 'T':
            return 1
        elif label == '0' or label == 'F':
            return 0
        else:
            raise ValueError
        
    def load_from_resource_hub(self):
        data_paths = self.find_data_paths()
        self.load_from_files(data_paths)

    # Loads from a list of pairs of target usages
    def load_from_target_usages(self, target_usages : List[Union[Tuple[TargetUsage], List[TargetUsage], TargetUsageList]], labels):
        data = []
        for i, target_usage_pair in enumerate(target_usages):
            tu1, tu2 = target_usage_pair
            example = {'word': tu1.text()[tu1.offsets[0]:tu1.offsets[1]],
                       'text1': tu1.text(),
                       'text2': tu2.text(),
                       'start1': tu1.offsets[0],
                       'end1': tu1.offsets[1],
                       'start2': tu2.offsets[0],
                       'end2': tu2.offsets[1],
                       'label': labels[i]}
            data.append(example)
        self.load_from_data(data)

    
    def evaluate(self, predictions : Union[List[Dict], Dict], dataset, metric : Callable, word = None):
        """
            Evaluates predictions by comparing them to the true labels of the dataset.
            Args:
                predictions (Union[List[Dict], Dict]) : the predictions. If a dict, id:s are expected in both this dict and the dataset to compare against.
                dataset (str) : one of ['train','dev','test','dev_larger',...]
                metric (Callable) : a metric such as scipy.stats.spearmanr, that can be used to compare the predictions.
        """
        dataset = self.get_dataset(dataset)

        if word is not None:
            if not word in self.target_words:
                logging.error(f'Word {word} was not found.')
                raise ValueError
            dataset = filter(lambda d : d['word'] == word, dataset)

        if type(predictions) == dict and 'id' in predictions.keys():
            for d in dataset:
                if not 'id' in d.keys():
                    logging.error('Could not find id:s for all examples in the dataset.')
                    raise KeyError
            pred = [predictions[ex['id']] for ex in dataset]
        else:
            pred = predictions
        truth = [ex['label'] for ex in dataset]
        try:
            stats = metric(truth, pred)
            return stats
        except:
            logging.error(f'Could not use {metric} to compare the true and predicted labels.')
    
    def evaluate_spearman(self, predictions : Union[List[Dict], Dict], dataset = 'test', word = None):
        return self.evaluate(predictions, dataset, spearmanr, word)

    def evaluate_accuracy(self, predictions : Union[List[Dict], Dict], dataset = 'test', word = None):
        return self.evaluate(predictions, dataset, accuracy_score, word)
    
    def evaluate_f1(self, predictions : Union[List[Dict], Dict], dataset = 'test', word = None, average='macro'):
        return self.evaluate(predictions, dataset, lambda truth, pred : f1_score(truth, pred, average=average), word)
    

class WSD(Benchmark):
    """
        Dataset handling for the Word Sense Disambiguation (WSD) task.
        Parameters:
            path (str) : a path to the dataset, if it is not stored by the resource hub in the cache folder.
            dataset (str|list|dict) : the dataset to be loaded. 'XL-WSD' if using a dataset from the language change resource hub, or a list or a dict if loading from a datastructure already describing a WSD dataset.
            version (str) : the version of the dataset if using a dataset in the resource hub.
            language (str) : the language code (e.g. BG).
            name (str) : the name of the dataset (in case no values for dataset, language and version are specified).
    """
    def __init__(self, 
                 path : str = None,
                 wsd_data : list | dict = None,
                 dataset : str = None, 
                 language : str = None, 
                 version : str = None,
                 subset : str = None,
                 name : str = None):
        self.data = {}
        self.target_words = set()
        self.dataset = dataset
        self.version = version
        self.language = language
        self.subset = subset
        if name is not None:
            self.name = name

        # Loads from the resource hub or a local path containing the necessary files
        if wsd_data == None and dataset != None and language != None and version != None:
            # Let the resource manager find the path
            if path == None:
                lc = LanguageChange()
                path = lc.get_resource('benchmarks', 'WSD', dataset, version)

                if dataset == 'XL-WSD':
                    self.home_path = os.path.join(path, 'xl-wsd')
            # Pre-defined path
            else:
                self.home_path = path

            logging.info(f"WSD home path: {self.home_path}")
            self.load(dataset, language)

        # Loads from a dictionary or list already containing a WSD dataset
        elif wsd_data != None:
            try:
                self.load_from_data(wsd_data)
            except Exception as e:
                logging.error('Could not load from dataset.')
                raise e

    # Loads from a dict or list containing a WSD dataset, with each example as in self.load()
    def load_from_data(self, data):
        if type(data) == list:
            self.data = {'all': data}
        elif type(data) == dict:
            self.data = data

        # Add the lemma in each data example to the set of target words
        for dataset in self.data.values():
            for d in dataset:
                if 'word' in d.keys():
                    self.target_words.add(d['word'])

    # Finds the file paths of the data and labels for possible train, dev and test sets.
    def find_data_paths(self, dataset, language):
        
        train_paths = {'data': None, 'labels': None}
        dev_paths= {'data': None, 'labels': None}
        test_paths = {'data': None, 'labels': None}
        data_paths = {'train': train_paths, 'dev': dev_paths, 'test': test_paths}

        if dataset == 'XL-WSD':

            if os.path.exists(os.path.join(self.home_path, f'training_datasets/semcor_{language.lower()}')):
                data_paths['train']['data'] = f'training_datasets/semcor_{language.lower()}/semcor_{language.lower()}.data.xml'
                data_paths['train']['labels'] = f'training_datasets/semcor_{language.lower()}/semcor_{language.lower()}.gold.key.txt'
            else:
                logging.info(f'No train set found for {language}.')
            
            if os.path.exists(os.path.join(self.home_path, f'evaluation_datasets/dev-{language.lower()}')):
                data_paths['dev']['data'] = f'evaluation_datasets/dev-{language.lower()}/dev-{language.lower()}.data.xml'
                data_paths['dev']['labels'] = f'evaluation_datasets/dev-{language.lower()}/dev-{language.lower()}.gold.key.txt'
            else:
                logging.info(f'No dev set found for {language}. Did you enter the right language code?')

            if os.path.exists(os.path.join(self.home_path, f'evaluation_datasets/test-{language.lower()}')):
                data_paths['test']['data'] = f'evaluation_datasets/test-{language.lower()}/test-{language.lower()}.data.xml'
                data_paths['test']['labels'] = f'evaluation_datasets/test-{language.lower()}/test-{language.lower()}.gold.key.txt'
            else:
                logging.info(f'No test set found for {language}. Did you enter the right language code?')

        else:
            logging.info("No data was found.")

        return data_paths
    
    # Reads an XML containing WSD data excl. labels
    def read_xml(self, path):
        data = []

        parser = ET.iterparse(path, events=('start', 'end'))

        sentence_tag = 'sentence'
        word_tag = 'wf'
        target_tag = 'instance'

        for event, elem in parser:

            if elem.tag == sentence_tag and event == 'start':
                sent = {'text': [], 'target_words': {}}

            elif elem.tag == word_tag and event == 'end':
                sent['text'].append(elem.text)

            elif elem.tag == target_tag and event == 'end':
                sent['text'].append(elem.text)
                sent['target_words'][elem.attrib['id']] = {'lemma': elem.attrib['lemma'], 'index': len(sent['text']) - 1}
                self.target_words.add(elem.attrib['lemma'])

            elif elem.tag == sentence_tag and event == 'end':
                data.append(sent)
                elem.clear()

        return data

    def load_from_files(self, data_paths, dataset):
        """
            Loads a dataset from paths to train, dev and test sets (possibly None).

            Args:
                data_paths (Dict[Dict[str, str],str]): a dictionary containing the paths to the different parts of the dataset, formatted as in self.find_data_paths().
                dataset (str): the name of the dataset.
        """
        data = {'train':[], 'dev':[], 'test':[]}
        
        if dataset == 'XL-WSD':
            for key in data_paths.keys():

                data_by_id = {}

                if data_paths[key]['data'] is not None:
                    raw_data = self.read_xml(os.path.join(self.home_path, data_paths[key]['data']))
                    for d in raw_data:
                        for id, target in d['target_words'].items():
                            start, end = self.word_index_to_char_indices(d['text'], target['index'])
                            data_by_id[id] = {'text': " ".join(d['text']), 'word': target['lemma'], 'start': start, 'end': end}

                if data_paths[key]['labels'] is not None:
                    with open(os.path.join(self.home_path, data_paths[key]['labels'])) as f:
                        for line in f:
                            line_data = line.strip("\n").split(" ")
                            id = line_data[0]
                            labels = line_data[1:]
                            if len(labels) > 1:
                                # If there are multiple lables, create new examples for each.
                                for i, label in enumerate(labels):
                                    data_by_id[f'{id}:{i}'] = data_by_id[id]
                                    data_by_id[f'{id}:{i}']['label'] = label
                                    data_by_id[f'{id}:{i}']['id'] = f'{id}:{i}'
                                del data_by_id[id]
                            else:
                                data_by_id[id]['label'] = labels[0]
                                data_by_id[id]['id'] = id

                data[key] = list(data_by_id.values())

        self.load_from_data(data)

    def load(self, dataset, language):
        data_paths = self.find_data_paths(dataset, language)
        self.load_from_files(data_paths, dataset)

    # Loads from a list of target usages
    def load_from_target_usages(self, target_usages : Union[List[TargetUsage], TargetUsageList], labels):
        data = []
        for i, tu in enumerate(target_usages):
            example = {'word': tu.text()[tu.offsets[0]:tu.offsets[1]],
                       'text': tu.text(),
                       'start': tu.offsets[0],
                       'end': tu.offsets[1],
                       'label': labels[i]}
            if hasattr(tu, 'id'):
                example['id'] = tu.id
            data.append(example)
        self.load_from_data(data)

    # Casts to a WSI object with the same data
    def cast_to_WSI(self):
        wsi = WSI(wsi_data = self.data)
        wsi.dataset = self.dataset
        wsi.version = self.version
        wsi.language = self.language
        if hasattr(self, 'name'):
            wsi.name = self.name
        #wsi.load_from_data(self.data)
        return wsi

    def evaluate(self, predictions : Union[List[Dict], Dict], dataset, metric, word = None):
        """
            Evaluates predictions by comparing them to the true labels of the dataset.
            Args:
                predictions (Union[List[Dict], Dict]) : the predictions. If a dict, id:s are expected in both this dict and the dataset to compare against.
                dataset (str) : one of ['train','dev','test','dev_larger',...]
                metric (Union[Callable, str]) : a metric such as scipy.stats.spearmanr, that can be used to compare the predictions. Either a function or a string to which there is a function associated.
        """
        dataset = self.get_dataset(dataset)

        if word is not None:
            if not word in self.target_words:
                logging.error(f'Word {word} was not found.')
                raise ValueError
            dataset = filter(lambda d : d['word'] == word, dataset)

        if type(predictions) == dict and 'id' in predictions.keys():
            for d in dataset:
                if not 'id' in d.keys():
                    logging.error('Could not find id:s for all examples in the dataset.')
                    raise KeyError
            pred = [predictions[ex['id']] for ex in dataset]
        else:
            pred = predictions
        truth = [ex['label'] for ex in dataset]

        metric_names = {'accuracy': accuracy_score, 'f1': f1_score}
        try:
            if type(metric) == str:
                try:
                    metric = metric_names[metric]
                except KeyError:
                    logging.error(f'Could not use {metric} as a metric.')
                    return
                
            stats = metric(truth, pred)
            return stats
        except:
            logging.error(f'Could not use {metric} to compare the true and predicted labels.')

    def evaluate_accuracy(self, predictions : Union[List[Dict], Dict], dataset = 'test', word = None):
        return self.evaluate(predictions, dataset, accuracy_score, word)
    
    def evaluate_f1(self, predictions : Union[List[Dict], Dict], dataset = 'test', word = None, average='macro'):
        return self.evaluate(predictions, dataset, lambda truth, pred : f1_score(truth, pred, average=average), word)


class WSI(Benchmark):
    """
        Dataset handling for the Word Sense Induction (WSI) task.
        Parameters:
            dataset (list|dict) : a datastructure describing a WSI dataset.
            name (str) : the name of the dataset (optional but useful for evaluation pipelines).
    """
    def __init__(self, 
                 wsi_data : list | dict = None,
                 dataset : str = None,
                 version : str = None,
                 language : str = None,
                 subset : str = None,
                 name : str = None):
        self.data = {}
        self.dataset = dataset
        self.version = version
        self.language = language
        self.subset = subset
        self.target_words = set()
        if name is not None:
            self.name = name
        if wsi_data is not None:
            self.load_from_data(wsi_data)
        else:
            logging.error('No data was specified.')
            raise ValueError

    # Loads from a list or dict containing a WSI dataset
    def load_from_data(self, data):
        if type(data) == list:
            self.data = {'all': data}
        elif type(data) == dict:
            self.data = data
        else:
            logging.error('Could not load the dataset as a WSI dataset.')
            raise TypeError

        # Add the lemma in each data example to the set of target words
        for dataset in self.data.values():
            for d in dataset:
                if 'word' in d.keys():
                    self.target_words.add(d['word'])

    # Loads from a list of target usages
    def load_from_target_usages(self, target_usages : Union[List[TargetUsage], TargetUsageList], labels):
        data = []
        for i, tu in enumerate(target_usages):
            example = {'word': tu.text()[tu.offsets[0]:tu.offsets[1]],
                       'text': tu.text(),
                       'start': tu.offsets[0],
                       'end': tu.offsets[1],
                       'label': labels[i]}
            if hasattr(tu, 'id'):
                example['id'] = tu.id
            data.append(example)
        self.load_from_data(data)

    def evaluate(self, predictions, metrics = {'ari','purity'}, dataset = 'all', average = False):
        """
            Evaluates a clustering with respect to the true labels as given in self.data.

            Args:
                predictions ({str: str|int}|[str|int]): a clustering as either a dictionary {id: cluster} or list .[cluster] of usage assignments. If it is a list, it is expected to be in the same order as the dataset evaluated on.
                metric (function|str): the metric to use for evaluation, such as RI, ARI or purity.
                dataset (str): the sub-dataset to use, e.g. 'test' or 'all'.

            Returns:
                scores ({str: float}): the score for each word.
        """
        labels_per_word = {}
        if type(predictions) == dict:
            for d in self.get_dataset(dataset):
                if not d['word'] in labels_per_word:
                    labels_per_word[d['word']] = [[],[]]
                labels_per_word[d['word']][0].append(d['label'])
                labels_per_word[d['word']][1].append(predictions[d['id']])

        elif type(predictions) == list or type(predictions) == np.ndarray:
            for i, d in enumerate(self.get_dataset(dataset)):
                if not d['word'] in labels_per_word:
                    labels_per_word[d['word']] = [[],[]]
                labels_per_word[d['word']][0].append(d['label'])
                labels_per_word[d['word']][1].append(predictions[i])

        metric_names = {'ari': adjusted_rand_score, 'purity': purity}
        reverse_metric_names = {v : k for (k, v) in metric_names.items()}
        metric_functions = set()
        for metric in metrics:
            if type(metric) == str:
                try:
                    metric = metric_names[metric]
                except KeyError:
                    logging.error(f'{metric} does not define a metric.')
                    continue
            metric_functions.add(metric)

        scores = {}
        for metric in metric_functions:
            if metric in reverse_metric_names:
                scores[reverse_metric_names[metric]] = {}
            if metric in reverse_metric_names:
                if not average:
                    for word, labels in labels_per_word.items():
                        gold_labels, pred_labels = labels
                        scores[reverse_metric_names[metric]][word] = metric(gold_labels, pred_labels)
                else:
                    scores[reverse_metric_names[metric]] = np.mean(list(metric(gold_labels, pred_labels) for gold_labels, pred_labels in labels_per_word.values()))

        return scores

    def evaluate_ari(self, predictions, dataset = 'all', average = False):
        return self.evaluate(predictions, {adjusted_rand_score}, dataset, average)
    
    def evaluate_purity(self, predictions, dataset = 'all', average = False):
        return self.evaluate(predictions, {purity}, dataset, average)