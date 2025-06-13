from languagechange.resource_manager import LanguageChange
from languagechange.corpora import LinebyLineCorpus
from languagechange.usages import Target, TargetUsageList, DWUGUsage
from languagechange.utils import NumericalTime, LiteralTime
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
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from typing import List, Dict


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
            for dataset in self.data.values():
                all_data += dataset
            return all_data
        
    def split_train_dev_test(self, train_prop = 0.8, dev_prop = 0.1, test_prop = 0.1, shuffle = True):
        for s in ['train','dev','test']:
            if s in self.data.keys():
                logging.info(f'Dataset already contains a {s} set.')
        if not 'all' in self.data.keys():
            self.data['all'] = []
            for dataset in self.data.values():
                self.data['all'].extend(dataset)
        
        assert train_prop + dev_prop + test_prop == 1

        if shuffle:
            random.shuffle(self.data['all'])
            
        train_offset = int(len(self.data['all']) * train_prop)
        dev_offset = train_offset + int(len(self.data['all']) * dev_prop)

        self.data['train'] = self.data['all'][:train_offset]
        self.data['dev'] = self.data['all'][train_offset:dev_offset]
        self.data['test'] = self.data['all'][dev_offset:]

    def get_data_by_word(self, dataset, word):
        dataset = self.get_dataset(dataset)
        return list(filter(lambda d : d['word'] == word, dataset))


class SemEval2020Task1(Benchmark):

    def __init__(self, language):
        lc = LanguageChange()
        self.language = language
        home_path = lc.get_resource('benchmarks', 'SemEval 2020 Task 1', self.language, 'no-version')
        semeval_folder = os.listdir(home_path)[0]
        self.home_path = os.path.join(home_path,semeval_folder)
        self.load()


    def load(self):
        self.corpus1_lemma = LinebyLineCorpus(os.path.join(self.home_path, 'corpus1', 'lemma'), name='corpus1_lemma', language=self.language, time=NumericalTime(1), is_lemmatized=True)
        self.corpus2_lemma = LinebyLineCorpus(os.path.join(self.home_path, 'corpus2', 'lemma'), name='corpus2_lemma', language=self.language, time=NumericalTime(2), is_lemmatized=True)
        self.corpus1_token = LinebyLineCorpus(os.path.join(self.home_path, 'corpus1', 'token'), name='corpus1_token', language=self.language, time=NumericalTime(1), is_tokenized=True)
        self.corpus2_token = LinebyLineCorpus(os.path.join(self.home_path, 'corpus2', 'token'), name='corpus2_token', language=self.language, time=NumericalTime(2), is_tokenized=True)
        self.binary_task = {}
        self.graded_task = {}

        with open(os.path.join(self.home_path, 'truth', 'binary.txt')) as f:
            for line in f:
                word, label = line.split()
                word = Target(word)
                self.binary_task[word] = int(label)

        with open(os.path.join(self.home_path, 'truth', 'graded.txt')) as f:
            for line in f:
                word, score = line.split()
                word = Target(word)
                self.graded_task[word] = float(score)


# Gets the character offsets from a tokenized sentence string and an index of the word in question.
def word_index_to_char_indices(text, word_index):
    split_text = text.split(" ")
    start = sum(len(s)+1 for s in split_text[:word_index])
    end = start + len(split_text[word_index])
    return start, end

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


class DWUG(Benchmark):

    def __init__(self, path=None, language=None, version=None):
        lc = LanguageChange()
        if not language == None and not version == None:
            self.language = language
            home_path = lc.get_resource('benchmarks', 'DWUG', self.language, version)
            dwug_folder = os.listdir(home_path)[0]
            self.home_path = os.path.join(home_path,dwug_folder)
        else:
            if not path == None and os.path.exists(path):
                self.home_path = path
            else:
                raise Exception('The path is None or does not exists.')       
        self.load()

    def load(self, config=None):
        self.target_words = os.listdir(os.path.join(self.home_path,'data'))
        self.stats_groupings = {}
        self.stats = {}

        stats_path = None
        if not config == None:
            stats_path = os.path.join(self.home_path,'stats',config)
        elif os.path.exists(os.path.join(self.home_path,'stats','opt')):
            stats_path = os.path.join(self.home_path,'stats','opt')
        else:
            stats_path = os.path.join(self.home_path,'stats')

        with open(os.path.join(stats_path,'stats_groupings.csv')) as f:
            keys = []
            for j,line in enumerate(f):
                line = line.replace('\n','').split('\t')
                if j > 0:
                    values = line
                    D = {keys[j]:values[j] for j in range(1,len(values))}
                    self.stats_groupings[values[0]] = D
                else:
                    keys = line

        with open(os.path.join(stats_path,'stats.csv')) as f:
            keys = []
            for j,line in enumerate(f):
                line = line.replace('\n','').split('\t')
                if j > 0:
                    values = line
                    D = {keys[j]:values[j] for j in range(1,len(values))}
                    self.stats[values[0]] = D
                else:
                    keys = line

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
        with open(os.path.join(self.home_path,'data',word,'uses.csv')) as f:
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

    def get_word_annotations(self, word):
        usages = TargetUsageList()
        with open(os.path.join(self.home_path,'data',word,'uses.csv')) as f:
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

    def get_stats(self):
        return self.stats

    def get_stats_groupings(self):
        return self.get_stats_groupings
    
    def cast_to_WiC(self, only_between_groups = False, remove_outliers = True, exclude_non_judgments = True, transform_labels = None):
        """
            Casts the DWUG to a Word in Context (WiC) dataset.

            Args:
                only_between_groups (bool) : if true, select only examples where the two usages belong to different groupings.
        """
        wic = WiC(language=self.language)
        data = []
        for word in self.stats_groupings:
            excluded_instances = set()

            with open(os.path.join(self.home_path,'clusters/opt',f'{word}.csv')) as f:
                for line in islice(f, 1, None):
                    line = line.replace('\n','').split('\t')
                    id, label = line
                    if remove_outliers and int(label) == -1:
                        excluded_instances.add(id)

            usages_by_key = {}
            with open(os.path.join(self.home_path,'data',word,'uses.csv')) as f:
                for line in islice(f, 1, None):
                    line = line.replace('\n','').split('\t')
                    lemma = line[0]
                    grouping = line[3]
                    id = line[4]
                    if not id in excluded_instances:
                        context_tokenized = line[9]
                        word_index = int(line[10])
                        start, end = wic.get_start_end(context_tokenized, word_index)
                        usages_by_key[id] = {'word': lemma, 'text':context_tokenized, 'start':start, 'end':end, 'grouping':grouping}

            temp_labels = {}
            with open(os.path.join(self.home_path,'data',word,'judgments.csv')) as f:
                for j,line in enumerate(f):
                    if not j == 0:
                        line = line.replace('\n','').split('\t')
                        idx1, idx2 = line[0], line[1]
                        label = int(line[3])
                        if (label != 0 or not exclude_non_judgments) and not idx1 in excluded_instances and not idx2 in excluded_instances:
                            if not frozenset([idx1,idx2]) in temp_labels:
                                temp_labels[frozenset([idx1,idx2])] = []
                            temp_labels[frozenset([idx1,idx2])].append(label)

            for key in temp_labels:
                ordered_ids = list(key)
                id1, id2 = ordered_ids[0], ordered_ids[1]
                usage1, usage2 = usages_by_key[id1], usages_by_key[id2]
                word = usage1['word']
                assert word == usage2['word']
                if only_between_groups and usage1['grouping'] == usage2['grouping']:
                    continue
                if transform_labels is not None:
                    try:
                        label = transform_labels(temp_labels[key])
                    except:
                        logging.error(f'{transform_labels} could not be used to transform labels.')
                        raise ValueError
                else:
                    label = np.mean(temp_labels[key])
                data.append({'word': word, 
                            'id1': id1, 'text1': usage1['text'], 'start1': usage1['start'], 'end1': usage1['end'],
                            'id2': id2, 'text2': usage2['text'], 'start2': usage2['start'], 'end2': usage2['end'],
                            'label': label})

        wic.load_from_data(data) 
        for d in data:
            wic.words.add(d['word'])
        return wic
    
    def get_usages_and_senses(self, remove_outliers = True): #So far this is very similar to the cast_to_WSD function
        data = []
        for word in self.stats_groupings:
            usages_by_id = {}
            with open(os.path.join(self.home_path,'clusters/opt',f'{word}.csv')) as f:
                for line in islice(f, 1, None):
                    line = line.replace('\n','').split('\t')
                    id, label = line
                    if not remove_outliers or int(label) != -1:
                        usages_by_id[id] = {'id': id, 'label': label}
            
            with open(os.path.join(self.home_path,'data',word,'uses.csv')) as f:
                for line in islice(f, 1, None):
                    line = line.replace('\n','').split('\t')
                    lemma = line[0]
                    id = line[4]
                    if id in usages_by_id:
                        context_tokenized = line[9]
                        word_index = int(line[10])
                        start, end = word_index_to_char_indices(context_tokenized, word_index)
                        usages_by_id[id].update({'word': lemma, 'text':context_tokenized, 'start':start, 'end':end, 'label': lemma + ":" + usages_by_id[id]['label']})
            data.extend(list(usages_by_id.values()))
        return data
        
    def cast_to_WSD(self, remove_outliers = True):
        data = self.get_usages_and_senses(remove_outliers)
        wsd = WSD()
        wsd.load_from_data(data)
        for d in data:
            wsd.target_words.add(d['word'])
        return wsd

    def cast_to_WSI(self, remove_outliers = True):
        data = self.get_usages_and_senses(remove_outliers)
        wsi = WSI()
        wsi.load_from_data(data)
        for d in data:
            wsi.target_words.add(d['word'])
        return wsi
    
    def cluster_evaluation(self, predictions, metrics = {'ari', 'purity'}, remove_outliers = True):
        wsi = self.cast_to_WSI(remove_outliers)
        return wsi.evaluate(predictions, metrics)
        

# Dataset handling for the Word-in-Context (WiC) task
class WiC(Benchmark):
    """
        Dataset handling for the Word-in-Context (WiC) task.
        Parameters:
            dataset (str) : the dataset to be loaded. One of ['WiC', 'XL-WiC', 'TempoWiC', 'MCL-WiC', 'AM2iCo'] if
                downloading from the language change resource hub, or empty if loading manually.
            version (str) : the version of the dataset if loading from the resource hub.
            language (str) : the language code (e.g. AR), if loading a multi- or crosslingual dataset from the resource hub.
            crosslingual (bool) : whether to use the crosslingual or multilingual dataset, in the case of MCL-WiC.
    """
    def __init__(self, dataset=None, version=None, language=None, crosslingual = False):
        self.data = {}
        self.dataset = dataset
        self.version = version
        self.language = language
        self.words = set()

        if dataset != None and version != None and (dataset == 'WiC' or dataset == 'TempoWiC' or language != None):
            lc = LanguageChange()
            home_path = lc.get_resource('benchmarks', 'WiC', dataset, version)
            
            if dataset == 'XL-WiC' or dataset == 'TempoWiC' or dataset == 'MCL-WiC' or dataset == 'AM2iCo':
                wic_folder = os.listdir(home_path)[0]
                home_path = os.path.join(home_path, wic_folder)
                if dataset == 'MCL-WiC':
                    if os.path.exists(os.path.join(home_path, "SemEval-2021_MCL-WiC_all-datasets.zip")):
                        with zipfile.ZipFile(os.path.join(home_path, "SemEval-2021_MCL-WiC_all-datasets.zip"), 'r') as f:
                            f.extractall(home_path)
                    self.home_path = os.path.join(home_path, 'MCL-WiC')
                else:
                    self.home_path = home_path
            else:
                self.home_path = home_path
            
            self.load_from_resource_hub(dataset, language, crosslingual = crosslingual)

    # Gets the character offsets from a tokenized sentence string and an index of the word in question.
    def get_start_end(self, text, word_index):
        split_text = text.split(" ")
        start = sum(len(s)+1 for s in split_text[:word_index])
        end = start + len(split_text[word_index])
        return start, end
    
    # Loads already formatted data, with each example as in self.load()
    def load_from_data(self, data):
        if type(data) == list:
            self.data = {'all': data}
        elif type(data) == dict:
            self.data = data

    # Finds the file paths of the data and labels for possible train, dev and test sets.
    def find_data_paths(self, dataset, language, crosslingual = False):
        train_paths = {'data':None, 'labels':None}
        dev_paths= {'data':None, 'labels':None}
        test_paths = {'data':None, 'labels':None}
        data_paths = {'train':train_paths, 'dev':dev_paths, 'test':test_paths}

        if dataset == 'WiC':
            for s in data_paths.keys():
                data_paths[s]['data'] = s + "/" + s + ".data.txt"
                data_paths[s]['labels'] = s + "/" + s + ".gold.txt"

        elif dataset == 'XL-WiC':
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
                language_path = language_paths[language]
            except KeyError:
                logging.error(f'Language {language} is not supported.')
                raise Exception

            # For English, train and dev sets are available, with both having labels.
            if language == 'EN':
                data_paths['train']['data'] = os.path.join(language_path, "train_en.txt")
                data_paths['dev']['data'] = os.path.join(language_path, "valid_en.txt")

            else:
                if os.path.exists(os.path.join(self.home_path, language_path, language.lower()+"_train.txt")):
                    data_paths['train']['data'] = os.path.join(language_path, language.lower()+"_train.txt")
                if os.path.exists(os.path.join(self.home_path, language_path, language.lower()+"_valid.txt")):
                    data_paths['dev']['data'] = os.path.join(language_path, language.lower()+"_valid.txt")
                if os.path.exists(os.path.join(self.home_path, language_path, language.lower()+"_test_data.txt")):
                    data_paths['test']['data'] = os.path.join(language_path, language.lower()+"_test_data.txt")
                if os.path.exists(os.path.join(self.home_path, language_path, language.lower()+"_test_gold.txt")):
                    data_paths['test']['labels'] = os.path.join(language_path, language.lower()+"_test_gold.txt")
    
        elif dataset == 'TempoWiC':
            data_paths['train']['data'] = "data/train.data.jl"
            data_paths['train']['labels'] = "data/train.labels.tsv"
            data_paths['dev']['data'] = "data/validation.data.jl"
            data_paths['dev']['labels'] = "data/validation.labels.tsv"
            data_paths['test']['data'] = "data/test-codalab-10k.data.jl"
            data_paths['test']['labels'] = "data/test.gold.tsv"

        elif dataset == 'MCL-WiC':
            # The multilingual task
            if not crosslingual:
                if language == 'en':
                    data_paths['train']['data'] = "training/training.en-en.data"
                    data_paths['train']['labels'] = "training/training.en-en.gold"
                data_paths['dev']['data'] = f"dev/multilingual/dev.{language.lower()}-{language.lower()}.data"
                data_paths['dev']['labels'] = f"dev/multilingual/dev.{language.lower()}-{language.lower()}.gold"
                data_paths['test']['data'] = f"test/multilingual/test.{language.lower()}-{language.lower()}.data"
            # The crosslingual task
            elif crosslingual:
                data_paths['test']['data'] = f"test/crosslingual/test.en-{language.lower()}.data"

        elif dataset == 'AM2iCo':

            language_path = "data/" + language.lower()
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
            index_to_offsets = self.get_start_end

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

    def load_from_files(self, data_paths, dataset, language = None, crosslingual = False):
        data = {'train':[], 'dev':[], 'test':[]}

        # The original Word-in-Context dataset
        if dataset == 'WiC':
            for key in data_paths.keys():
                if data_paths[key]['data'] is not None:
                    data[key] = self.load_from_txt(data_paths[key]['data'], word_indexes=True, field_map={'word':0,'indexes':2, 'text1':3, 'text2':4})
                if data_paths[key]['labels'] is not None:
                    labels = self.load_from_txt(data_paths[key]['labels'], field_map={'label': 0})
                    data[key] = [d | labels[i] for i, d in enumerate(data[key])]

        # XL-WiC, containing WiC datasets for 12 more languages other than English.
        elif dataset == 'XL-WiC':
            # There is something unusual in the offsets of the XL-WiC dataset for Farsi.
            # The first index is for some reason a word index while the difference between the first and the second denotes
            # the length of the word.
            if language == 'FA':
                def get_start_end(text, start, end):
                    char_start = sum(len(s)+1 for s in text.split(" ")[:int(start)])
                    char_end = char_start + int(end) - int(start)
                    return char_start, char_end
                
            for key in ['train','dev']:
                if data_paths[key]['data'] is not None:
                    data[key] = self.load_from_txt(data_paths[key]['data'])
                    if language == 'FA':
                        for i, d in enumerate(data[key]):
                            data[key][i]['start1'], data[key][i]['end1'] = get_start_end(d['text1'], d['start1'], d['end1'])
                            data[key][i]['start2'], data[key][i]['end2'] = get_start_end(d['text2'], d['start2'], d['end2'])

            if data_paths['test']['data'] is not None:
                data['test'] = self.load_from_txt(data_paths['test']['data'])
                if language == 'FA':
                    for i, d in enumerate(data['test']):
                        data['test'][i]['start1'], data['test'][i]['end1'] = get_start_end(d['text1'], d['start1'], d['end1'])
                        data['test'][i]['start2'], data['test'][i]['end2'] = get_start_end(d['text2'], d['start2'], d['end2'])  
            
                if data_paths['test']['labels'] is not None:
                    labels = self.load_from_txt(data_paths['test']['labels'], field_map={'label': 0})
                    data['test'] = [d | labels[i] for i, d in enumerate(data['test'])]

            

        # TempoWiC, containing social media data annotated with dates.
        elif dataset == "TempoWiC":
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
        elif dataset == "MCL-WiC":
            for key in data_paths.keys():
                data_dict = {}
                
                if data_paths[key]['data'] is not None:
                    with open(os.path.join(self.home_path, data_paths[key]['data'])) as f:
                        json_data = json.load(f)

                        for ex in json_data:
                            word = ex['lemma']
                            text1 = ex['sentence1']
                            text2 = ex['sentence2']

                            if not crosslingual:
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
        elif dataset == "AM2iCo":
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

        for key in data.keys():
            for d in data[key]:
                if 'word' in d:
                    self.words.add(d['word'])

        self.data = data
        
    def format_label(self, label):
        if label == 1 or label == 0:
            return label
        if label == '1' or label == 'T':
            return 1
        elif label == '0' or label == 'F':
            return 0
        else:
            raise ValueError
        
    def load_from_resource_hub(self, dataset, language, crosslingual = False):
        data_paths = self.find_data_paths(dataset, language, crosslingual)
        self.load_from_files(data_paths, dataset, language, crosslingual)
    
    def evaluate(self, predictions : List[Dict] | Dict, dataset, metric, word = None):
        """
            Evaluates predictions by comparing them to the true labels of the dataset.
            Args:
                predictions (list(dict) | dict) : the predictions. If a dict, id:s are expected in both this dict and the
                dataset to compare against.
                dataset (str) : one of ['train','dev','test','dev_larger',...]
                metric (function) : a metric such as scipy.stats.spearmanr, that can be used to compare the predictions
        """
        dataset = self.get_dataset(dataset)

        if word is not None:
            if not word in self.words:
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
    
    def evaluate_spearman(self, predictions : List[Dict] | Dict, dataset = 'test', word = None):
        return self.evaluate(predictions, dataset, spearmanr, word)

    def evaluate_accuracy(self, predictions : List[Dict] | Dict, dataset = 'test', word = None):
        return self.evaluate(predictions, dataset, accuracy_score, word)
    
    def evaluate_f1(self, predictions : List[Dict] | Dict, dataset = 'test', word = None, average='macro'):
        return self.evaluate(predictions, dataset, lambda truth, pred : f1_score(truth, pred, average=average), word)
    

# Dataset handling for the Word Sense Disambiguation (WSD) task (to be implemented)
class WSD(Benchmark):
    def __init__(self):
        super().__init__()


# Dataset handling for the Word Sense Induction (WSI) task (to be implemented)
class WSI(Benchmark):
    def __init__(self):
        self.data = {}
        self.target_words = set()

    # Loads already formatted data, with each example as in self.load()
    def load_from_data(self, data):
        if type(data) == list:
            self.data = {'all': data}
        elif type(data) == dict:
            self.data = data

    def evaluate(self, predictions, metrics = {'ari','purity'}, dataset = 'all'):
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

        elif type(predictions) == list:
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
            for word, labels in labels_per_word.items():
                gold_labels, pred_labels = labels
                if metric in reverse_metric_names:
                    scores[reverse_metric_names[metric]][word] = metric(gold_labels, pred_labels)

        return scores

    def evaluate_ari(self, predictions, dataset = 'all'):
        return self.evaluate(predictions, {adjusted_rand_score}, dataset)
    
    def evaluate_purity(self, predictions, dataset = 'all'):
        return self.evaluate(predictions, {purity}, dataset)
