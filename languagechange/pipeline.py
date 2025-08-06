from languagechange.models.representation.contextualized import ContextualizedModel
from languagechange.models.representation.definition import DefinitionGenerator, ChatModelDefinitionGenerator, LlamaDefinitionGenerator, T5DefinitionGenerator
from languagechange.models.representation.prompting import PromptModel
from languagechange.models.meaning.clustering import Clustering
from languagechange.usages import TargetUsage, TargetUsageList
from languagechange.models.change.metrics import GradedChange
from languagechange.benchmark import WiC, WSD, WSI, CD, SemEval2020Task1, DWUG
from pydantic import BaseModel, Field
from typing import List, Set, Union
import numpy as np
import json
import logging
import inspect
import os
from queue import Queue
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Utility function to update a dictionary recursively
def deep_update(d, u):
    for k, v in u.items():
        if type(v) == dict:
            if k in d:
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        else:
            d[k] = v
    return d


class Pipeline:
    def __init__(self):
        pass

    def save_evaluation_results(self, results, outfile, latex_path = None, decimals = None):
        if os.path.exists(outfile):
            with open(outfile, 'r+') as f:
                previous_results = json.load(f)
                results = deep_update(previous_results, results)
                f.seek(0)
                json.dump(results, f, indent=4)
                logging.info(f'Evaluation results saved to {outfile}') 
        else:
            with open(outfile, 'w') as f:
                json.dump(results, f, indent=4)
                logging.info(f'Evaluation results saved to {outfile}')
        if latex_path != None:
            self.generate_latex_table(results, latex_path, decimals)
            

    def generate_latex_table(self, data, save_path, decimals=None):
        """
            Generates a table of results in LaTeX format, to be saved in a .tex file. Meant to be used together with self.save_evaluation_results.
        """

        # Calculates the maximum depth of a nested dictionary
        def depth(d):
            if isinstance(d, dict):
                return 1 + max(depth(v) for v in d.values())
            return 1
        
        # Counts the amount of leaves for each sub-dictionary of a nested dictionary
        def count_leaves(data):
            tree = [0, []]
            # Find the amount of leaves in each subtree of d
            for v in data.values():
                # If we found the final {model: score} dict, return 1, as all models share 1 column in the table
                if type(v) != dict:
                    return 1
                subtree = count_leaves(v)
                tree[1].append(subtree)
                # A subtree which is more than a leaf
                if type(subtree) == list:
                    tree[0] += subtree[0]
                # A leaf
                else:
                    tree[0] += subtree
            return tree

        # Find the column width needed for each row in the table header, given previously calculated leaf counts
        def get_column_widths(tree, max_depth):

            # Each row is initialized as an empty queue
            columns = [Queue() for _ in range(max_depth)]

            # Recursively finds the width needed for all columns
            def get_cols_rec(cols, tree, depth):
                if type(tree) == list:
                    # Add the value in the right row
                    cols[depth].put(tree[0])
                    for subtree in tree[1]:
                        # Recursive call for all sub-columns
                        get_cols_rec(cols, subtree, depth + 1)
                else:
                    for d in range(depth, max_depth):
                        # Add 1s for the rest of the rows (for columns with shorter content than the maximum)
                        cols[d].put(tree)

            get_cols_rec(columns, tree, 0)

            # We don't keep the first row
            return columns[1:]

        # Prints columns in LaTeX format, given the original data with column names
        def get_table_rows(data, column_widths):
            table_rows = [[] for _ in range(len(column_widths) - 1)]

            scores = []
            models = set()

            def get_rows_rec(data, column_widths, depth):
                for k, v in data.items():
                    if type(v) == dict:
                        # Normal case
                        if sum(int(type(vv) == dict) for vv in v.values()) != 0:
                            table_rows[depth].append("\\multicolumn{" + str(column_widths[depth].get()) + "}{|c|}{" + k + "}")
                            get_rows_rec(v, column_widths, depth + 1)
                        # The case where we have reached the last row before the model and score, i.e. the row describing the metric.
                        else:
                            # Add the metric name to the last row
                            table_rows[-1].append("\\multicolumn{" + str(column_widths[depth].get()) + "}{|c|}{" + k + "}")
                            scores.append(v.copy())
                            for model in v.keys():
                                models.add(model)
                            for d in range(depth, len(table_rows) - 1):
                                table_rows[d].append("") # Add empty space to accommodate for longer columns
            
            get_rows_rec(data, column_widths, 0)

            return table_rows, scores, models

        max_depth = depth(data)

        tree = count_leaves(data)

        column_widths = get_column_widths(tree, max_depth)

        columns_str = "|c|"+"|".join(["c"] * tree[0])+"|"

        table_beginning = """\\begin{table}[h]
            \\centering
            \\begin{tabular}{"""+ columns_str +"}\hline"
            
        table_end = """
                \hline
            \\end{tabular}
        \\end{table}
            """

        table_rows, scores, models = get_table_rows(data, column_widths)

        # If decimals is provided, round each score to the amount of decimals set
        if decimals != None and type(decimals) == int:
            for model_scores in scores:
                for model, score in model_scores.items():
                    model_scores[model] = round(score, decimals)

        scores_per_model = {model: [s.get(model, '--') for s in scores] for model in models}

        print(scores_per_model)

        scores_string = "\\\\".join(model + "\t&" + "\t&".join(str(s) for s in scores) for model, scores in scores_per_model.items())

        table_string = table_beginning + "\\\\\n".join("\t&" + "\t&".join(row) for row in table_rows) + "\\\\\n" + scores_string + "\\\\\n" + table_end

        table_string = re.sub("_", "\_", table_string)

        # Save the LaTeX string to a .tex file
        if save_path.endswith(".tex"):
            with open(save_path,'w+') as f:
                f.write(table_string)
        else:
            raise Exception("The file needs to end in .tex")


class WSIPipeline(Pipeline):
    def __init__(self, dataset, usage_encoding, clustering, partition = 'test', shuffle = True, labels=[], dataset_name = None):
        super().__init__()
        if isinstance(dataset, WSI):
            self.dataset = dataset
        else:
            self.dataset = WSI(name=dataset_name) #TODO: exceptions
            self.dataset.load_from_target_usages(dataset, labels)
            self.dataset.split_train_dev_test(shuffle=shuffle)

        self.partition = partition
        self.evaluation_set = self.dataset.get_dataset(self.partition)
        self.usage_encoding = usage_encoding
        self.clustering = clustering

    def evaluate(self, return_labels = False, save = False, path = None):
        """
            Evaluate on the WSI task. Returns the ARI and purity scores and optionally the clustering labels.
        """
        target_usages = TargetUsageList()

        for example in self.evaluation_set:
            if 'id' in example:
                target_usages.append(TargetUsage(example['text'], [example['start'],example['end']], id=example['id']))
            else:
                target_usages.append(TargetUsage(example['text'], [example['start'],example['end']]))

        if isinstance(self.usage_encoding, DefinitionGenerator):
            encoded_usages = self.usage_encoding.generate_definitions(target_usages, encode_definitions = 'vectors')

        elif isinstance(self.usage_encoding, PromptModel):
            return None # Is this possible?
        
        elif isinstance(self.usage_encoding, ContextualizedModel):
            encoded_usages = self.usage_encoding.encode(target_usages)
            
        else:
            logging.error('Model not supported.')
            return None

        # Cluster the encoded usages
        clustering_results = self.clustering.get_cluster_results(encoded_usages)

        # Compute ARI and purity scores
        scores = self.dataset.evaluate(clustering_results.labels, dataset=self.partition)

        if save:
            if path == None:
                logging.error("Tried to save results but no path was specified.")
            else:

                if hasattr(self.dataset, 'name'):
                    self.save_evaluation_results({'WSI': {self.dataset.name: {metric: {type(self.usage_encoding).__name__: score} for metric, score in scores.items()}}}, path)

                elif hasattr(self.dataset, 'dataset') and self.dataset.dataset != None:
                    parameters = ['dataset', 'language', 'version']
                    dataset_info = {}
                    d = dataset_info
                    for param in parameters:
                        if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                            d[getattr(self.dataset, param)] = {}
                            d = d[getattr(self.dataset, param)]
                    for metric, score in scores.items():
                        d[metric] = {type(self.usage_encoding).__name__: score}
                    self.save_evaluation_results({'WSI': dataset_info}, path)

                else:
                    logging.error("Dataset has no 'name' attribute, nor 'dataset' attribute. Scores could therefore not be saved.")

        if return_labels:
            return scores, clustering_results.labels
        return scores

        
class WiCPipeline(Pipeline):
    def __init__(self, dataset, usage_encoding, partition = 'test', shuffle = True, labels=[], dataset_name = None):
        super().__init__()
        if isinstance(dataset, WiC):
            self.dataset = dataset
        else:
            self.dataset = WiC(name=dataset_name) #TODO: exceptions
            self.dataset.load_from_target_usages(dataset, labels)
            self.dataset.split_train_dev_test(shuffle=shuffle)

        self.partition = partition
        self.evaluation_set = self.dataset.get_dataset(self.partition)
        self.usage_encoding = usage_encoding

    def evaluate(self, task, label_func = None, save = False, path = None):
        """
            Evaluates on the WiC task. Returns accuracy and f1 scores if task='binary', Spearman correlation if task='graded'.
        """
        if task not in {'binary','graded'}:
            logging.error(f'Invalid argument for \'task\', should be one of [\'binary\', \'graded\']')
            return None
        
        labels = []

        if isinstance(self.usage_encoding, DefinitionGenerator) or isinstance(self.usage_encoding, ContextualizedModel):
            # Find the unique usages among all pairs
            index = dict() # Index to point to the right position in the usage/embeddings list when comparing usages in pairs
            i = 0
            usage_list = TargetUsageList()
            for pair in self.evaluation_set:
                for j in range(1,3):
                    if f'id{j}' in pair:
                        id = pair[f'id{j}']
                    else:
                        id = (pair[f'text{j}'], pair[f'start{j}'], pair[f'end{j}'])
                    if id not in index:
                        index[id] = i
                        i += 1
                        usage_list.append(TargetUsage(pair[f'text{j}'], [pair[f'start{j}'], pair[f'end{j}']]))

            if isinstance(self.usage_encoding, ContextualizedModel):
                encoded_usages = self.usage_encoding.encode(usage_list)

            elif isinstance(self.usage_encoding, DefinitionGenerator):
                encoded_usages = self.usage_encoding.generate_definitions(usage_list, encode_definitions = 'vectors')

            if label_func == None:
                if task == "graded":
                    label_func = lambda e1, e2 : np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2))
                elif task == "binary":
                    label_func = lambda e1, e2 : int(np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2)) > 0.5)
            
            elif callable(label_func):
                signature = inspect.signature(label_func)
                n_req_args = sum([int(p.default == p.empty) for p in signature.parameters.values()])
                if n_req_args != 2:
                    logging.error(f"'label_func' must take 2 arguments but takes {n_req_args}.")
                    return None
            else:
                logging.error("'label_func' must be a callable function.")
                return None

            for pair in self.evaluation_set:
                embedding_pair = []

                for j in range(1,3):
                    if f'id{j}' in pair:
                        id = pair[f'id1']
                    else:
                        id = (pair[f'text{j}'], pair[f'start{j}'], pair[f'end{j}'])
                    embedding_pair.append(encoded_usages[index[id]])

                labels.append(label_func(embedding_pair[0], embedding_pair[1]))

        elif isinstance(self.usage_encoding, PromptModel):
            if task == "graded":
                class WiCGraded(BaseModel):
                    change : float = Field(description='How similar the two occurrences of the word are.',le=1, ge=0)#perhaps rename change to something else
                self.usage_encoding.structure = WiCGraded
                for pair in self.evaluation_set:
                    target_usage_list = TargetUsageList([TargetUsage(pair['text1'], [pair['start1'], pair['end1']]),
                                                         TargetUsage(pair['text2'], [pair['start2'], pair['end2']])])
                    labels.append(self.usage_encoding.get_response(target_usage_list, user_prompt_template = 'Please tell me how similar the meaning of the word \'{target}\' is in the following example sentences: \n1. {usage_1}\n2. {usage_2}'))
            elif task == "binary":
                class WiCBinary(BaseModel):
                    change : bool = Field(description='Whether the word has the same meaning or not.')
                self.usage_encoding.structure = WiCBinary
                for pair in self.evaluation_set:
                    target_usage_list = TargetUsageList([TargetUsage(pair['text1'], [pair['start1'], pair['end1']]),
                                                         TargetUsage(pair['text2'], [pair['start2'], pair['end2']])])
                    label = int(self.usage_encoding.get_response(target_usage_list, user_prompt_template = 'Please tell me if the meaning of the word \'{target}\' is the same in the following example sentences: \n1. {usage_1}\n2. {usage_2}'))
                    labels.append(label)

        else:
            logging.error('Model not supported.')
            return None
        
        if task == 'binary':
            acc = self.dataset.evaluate_accuracy(labels, self.partition)
            f1 = self.dataset.evaluate_f1(labels, self.partition)
            scores = {'accuracy': acc, 'f1': f1}

        elif task == 'graded':
            spearman_r = self.dataset.evaluate_spearman(labels, self.partition)
            scores = {'spearman_r': spearman_r}

        if save:
            if path == None:
                logging.error("Tried to save results but no path was specified.")
            else:
                if hasattr(self.dataset, 'name'):
                    scores_dict = {'WiC': {self.dataset.name: {metric: {type(self.usage_encoding).__name__: score} for metric, score in scores.items()}}}
                    self.save_evaluation_results(scores_dict, path)

                elif hasattr(self.dataset, 'dataset') and self.dataset.dataset != None:
                    parameters = ['dataset', 'language', 'version', 'crosslingual']
                    dataset_info = {}
                    d = dataset_info
                    for param in parameters:
                        if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                            d[getattr(self.dataset, param)] = {}
                            d = d[getattr(self.dataset, param)]
                    for metric, score in scores.items():
                        d[metric] = {type(self.usage_encoding).__name__: score}
                    scores_dict = {'WiC': dataset_info}
                    self.save_evaluation_results(scores_dict, path)

                else:
                    logging.error("Dataset has no 'name' attribute, nor 'dataset' attribute. Scores could therefore not be saved.")
        
        return scores


class GCDPipeline(Pipeline):
    def __init__(self, dataset : Union[DWUG, List[Set[TargetUsage]]],
                 usage_encoding,
                 metric : GradedChange,
                 scores : List = None,
                 dataset_name : str = None):
        super().__init__()
        if isinstance(dataset, DWUG) or isinstance(dataset, SemEval2020Task1):
            self.dataset = dataset
        else:
            self.dataset = CD(name=dataset_name)
            self.dataset.load_from_target_usages(dataset, scores)

        self.usage_encoding = usage_encoding
        self.metric = metric

    def evaluate(self, save = False, path = None):
        """
            Evaluates on the GCD task. Returns the Spearman correlation between the predicted and ground truth change scores.
        """
        change_scores = {}

        if isinstance(self.usage_encoding, PromptModel):
            return None # Is this possible?
        
        elif isinstance(self.usage_encoding, DefinitionGenerator) or isinstance(self.usage_encoding, ContextualizedModel):
            if isinstance(self.dataset, SemEval2020Task1):
                target_usages_t1_all_words = self.dataset.corpus1_lemma.search([target.target for target in self.dataset.graded_task.keys()])
                target_usages_t2_all_words = self.dataset.corpus2_lemma.search([target.target for target in self.dataset.graded_task.keys()])

            for target in self.dataset.graded_task.keys():
                word = target.target

                if isinstance(self.dataset, DWUG):
                    target_usages = self.dataset.get_word_usages(word)
                    target_usages_t1 = [u for u in target_usages if u.grouping == "1" ]
                    target_usages_t2 = [u for u in target_usages if u.grouping == "2" ]

                elif isinstance(self.dataset, SemEval2020Task1):
                    target_usages_t1 = target_usages_t1_all_words[word]
                    target_usages_t2 = target_usages_t2_all_words[word]

                elif isinstance(self.dataset, CD):
                    target_usages_t1 = self.dataset.target_usages_t1[word]
                    target_usages_t2 = self.dataset.target_usages_t2[word]

                if isinstance(self.usage_encoding, DefinitionGenerator):
                    encoded_usages_t1 = self.usage_encoding.generate_definitions(target_usages_t1, encode_definitions='vectors')
                    encoded_usages_t2 = self.usage_encoding.generate_definitions(target_usages_t2, encode_definitions='vectors')

                elif isinstance(self.usage_encoding, ContextualizedModel):
                    encoded_usages_t1 = self.usage_encoding.encode(target_usages_t1)
                    encoded_usages_t2 = self.usage_encoding.encode(target_usages_t2)

                # Measure the change using the metric
                change = self.metric.compute_scores(encoded_usages_t1, encoded_usages_t2)

                change_scores[word] = change

        else:
            logging.error('Model not supported.')
            return None

        spearman_r = self.dataset.evaluate_gcd(change_scores)
        scores = {'spearman_r': spearman_r}

        if save:
                if path == None:
                    logging.error("Tried to save results but no path was specified.")
                else:
                    if hasattr(self.dataset, 'name'):
                        scores_dict = {'GCD': {self.dataset.name: {metric: {type(self.usage_encoding).__name__: score} for metric, score in scores.items()}}}
                        self.save_evaluation_results(scores_dict, path)

                    elif hasattr(self.dataset, 'dataset'):
                        parameters = ['dataset', 'language', 'version']
                        dataset_info = {}
                        d = dataset_info
                        for param in parameters:
                            if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                                d[getattr(self.dataset, param)] = {}
                                d = d[getattr(self.dataset, param)]
                        for metric, score in scores.items():
                            d[metric] = {type(self.usage_encoding).__name__: score}
                        scores_dict = {'GCD': dataset_info}
                        self.save_evaluation_results(scores_dict, path)

                    else:
                        logging.error("Dataset has no 'name' attribute, nor 'version' and 'language' attributes. Scores could therefore not be saved.")     

        return scores
