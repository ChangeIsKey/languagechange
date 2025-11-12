from languagechange.models.representation.contextualized import ContextualizedModel
from languagechange.models.representation.definition import DefinitionGenerator, ChatModelDefinitionGenerator, LlamaDefinitionGenerator, T5DefinitionGenerator
from languagechange.models.representation.prompting import PromptModel
from languagechange.models.meaning.clustering import Clustering, APosterioriaffinityPropagation
from languagechange.usages import TargetUsage, TargetUsageList
from languagechange.models.change.metrics import GradedChange, APD, PRT, PJSD
from languagechange.models.change.widid import WiDiD
from languagechange.benchmark import WiC, WSD, WSI, SemanticChangeEvaluationDataset, SemEval2020Task1, DWUG
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
            if k in d and type(d[k]) == dict:
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        else:
            d[k] = v
    return d


class Pipeline:
    def __init__(self):
        pass

    def save_evaluation_results(self, results, json_path : str = None, latex_path : str = None, decimals = None):
        if json_path != None:
            if os.path.exists(json_path):
                with open(json_path, 'r+') as f:
                    previous_results = json.load(f)
                    results = deep_update(previous_results, results)
                    f.seek(0)
                    json.dump(results, f, indent=4)
                    f.truncate()
                    logging.info(f'Evaluation results saved to {json_path}') 
            else:
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=4)
                    logging.info(f'Evaluation results saved to {json_path}')
        if latex_path != None:
            self.generate_latex_table(results, latex_path, decimals)
                
    def generate_latex_table(self, data, save_path, decimals=None, remove_headers=0, max_w=None, natural_split=False, remove_empty=False, sort_models=False, generate_caption=False, highlight_best=False):
        """
            Generates one or more tables of results in LaTeX format, to be saved in a .tex file. Meant to be used together with self.save_evaluation_results.
            Args:
                data (dict): the evaluation results, in a dictionary (similar to that produced by self.save_evaluation results).
                save_path (str): where to save the .tex file.
                decimals (int): the amount of decimals to round evaluation results to.
                remove_headers (int): if >0, the first remove_headers rows of the table are removed.
                max_w (int|List[int]): if not None, split the table into smaller tables. If an int, each table will be max_w columns wide (excluding the model names to the very left). If a list of ints, the value of max_w[i] is the width of table i.
                natural_split (int|bool): if True, split the table according to the first row which has a natural split. If an int >= 0, split the table according to the split of this row.
                remove_empty (bool): If True, remove all rows containing no value.
                sort_models (bool): If True, sort the rows alphabetically by the names of the models.
                generate_caption (bool): If True, generate a caption for each table.
                highlight_best (bool|Callable|str): If "max", highlight the highest value in each column. If "min", highlight the lowest value in each column. If a callable, use this as a function to compare values.
        """
        def get_column_widths(data):
            """
            Recursively computes the column widths per depth from a dict containing evaluation data.
            Returns a list of queues containing the widths of each column in each row, as well as the total amount of columns.
            """
            columns = [Queue()]

            def add_columns(d, depth=0):
                while len(columns) <= depth + 1:
                    columns.append(Queue())
                # Base case: reached a leaf (model: score)
                if not isinstance(d, dict) or not all(isinstance(v, dict) for v in d.values()):
                    for q in columns[depth:]:
                        q.put(1)
                    return 1
                # Recursive step: sum leaf counts of children
                total = 0
                for v in d.values():
                    total += add_columns(v, depth + 1)
                columns[depth].put(total)
                return total

            total_cols = add_columns(data)
            # Skip the first row
            return columns[1:], total_cols

        def get_table_rows(data, column_widths):
            """
                Joins the content of each entry with the width of the cell in the table.
                Merges empty cells together if they belong to the same supercolumn.
                Gets all the models and their scores in the right order.
            """
            table_rows = [[] for _ in range(len(column_widths) - 1)]
            scores = []
            models = set()

            def get_rows_rec(data, column_widths, depth):
                for k, v in data.items():
                    if type(v) == dict:
                        # Normal case
                        if sum(int(type(vv) == dict) for vv in v.values()) != 0:
                            w = column_widths[depth].get()
                            table_rows[depth].append((k,w))
                            # Recursive call to go further down the tree
                            get_rows_rec(v, column_widths, depth + 1)
                        # The case where we have reached the last row before the model and score, i.e. the row describing the metric.
                        else:
                            w = column_widths[depth].get()
                            # Add the metric name to the last row
                            table_rows[-1].append((k,w))
                            # Add the scores and models
                            scores.append(v.copy())
                            for model in v.keys():
                                models.add(model)
                            for d in range(depth, len(table_rows) - 1):
                                # Add empty space to accommodate for longer columns
                                table_rows[d].extend([('',1)] * w)

                # Merge empty cells together after each recursive call
                for d in range(depth, len(table_rows) - 1):
                    count = 0
                    row = []
                    for i, entry in enumerate(table_rows[d]):
                        # If we find an empty cell, increase the empty count
                        if entry == ('', 1):
                            count += 1
                            # If we reached the end of the row, add the empty space
                            if i == len(table_rows[d]) - 1:
                                row.append(('', count))
                        else:
                            # Add empty cells together
                            if count > 0:
                                row.append(('',count))
                                count = 0
                            # Add the current cell
                            row.append(entry)
                    # Update the row
                    table_rows[d] = row
            
            get_rows_rec(data, column_widths, 0)
            return table_rows, scores, models

        def split_header_row(row, n_cols : List[int]):
            """
                Splits a row into multiple rows, with row i n_cols[i] wide.
            """
            assert sum(n_cols) == sum(w for _, w in row), "The widths of split rows do not sum up to the total width"
            split_rows = [[] for _ in n_cols]
            s, curr_w = (None, 0)
            curr_split_row = 0
            w_left = n_cols[0]
            i = 0
            while curr_split_row < len(n_cols) and (i < len(row) or curr_w > 0):
                if curr_w == 0:
                    # Get a new entry from the row
                    s, curr_w = row[i]
                    i += 1
                w_to_add = min(curr_w, w_left)
                # Add the minimum of the width of the entry and the width left for the split row
                split_rows[curr_split_row].append((s, w_to_add))
                curr_w -= w_to_add
                w_left -= w_to_add
                # If there is now no space left, move on to the next split row
                if w_left == 0:
                    curr_split_row += 1
                    if curr_split_row < len(n_cols):
                        w_left = n_cols[curr_split_row]
            return split_rows
        
        def get_horizontal_lines(table_rows):
            """
                Draws horizontal lines between table rows where it fits.
            """
            line_strings = ["" for _ in range(len(table_rows))]
            for i, r in enumerate(table_rows[:-2]):
                index1 = 0
                for s1, w1 in r:
                    index2 = 0
                    match = False
                    for s2, w2 in table_rows[i+1]:
                        # If the two rows have matching multicolumns and one of them is empty, don't draw a horizontal line between them
                        if index2 == index1 and index2 + w2 == index1 + w1 and (s1 == '' or s2 == ''):
                            match = True
                        index2 += w2
                    if not match:
                        line_strings[i] += "\\cline{" + str(index1 + 2) + "-" + str(index1 + w1 + 1) + "}"
                    index1 += w1
            # Before the metrics, add a complete horizontal line
            line_strings[-2] = "\\cline{2-" + str(sum(w for _, w in table_rows[0])+1) + "}"
            return line_strings
        
        def render_header_row(row):
            return "\t&" + "\t&".join(["\\multicolumn{"+ str(w) + "}{|c|}{" + s + "}" for (s, w) in row]) + "\\\\"
        
        # Puts together the different parts of a table
        def create_table_string(table_rows, score_string):
            total_width = sum(w for _, w in table_rows[0])
            columns_str = "c|"+"|".join(["c"] * total_width)+"|"

            table_beginning = """
\\begin{table}[h]
    \\centering
    \\begin{tabular}{"""+ columns_str +"}\\cline{2-"+str(total_width+1)+"}"
                
            table_end = """
        \hline
    \\end{tabular}""" + (("\n\\caption{Evaluation results on the " + table_rows[0][0][0] + " task.}") if generate_caption else "") + """
\\end{table}"""
            
            line_strings = get_horizontal_lines(table_rows)
            header_string = "\n".join(render_header_row(row) + line_strings[i] for i, row in enumerate(table_rows))
            table_string = table_beginning + header_string + "\\hline\n" + score_string + "\\\\\n" + table_end
            table_string = re.sub("_", "\_", table_string)

            return table_string
        
        column_widths, total_cols = get_column_widths(data)
    
        table, scores, models = get_table_rows(data, column_widths)
        table = table[remove_headers:]

        # Split the table according to a natural subdivision in the data
        if natural_split:
            i = 0
            # If an int, use the natural split of this row
            if type(natural_split) == int:
                i = natural_split
            # Otherwise, choose the first row that has a split
            else:
                for j, row in enumerate(table):
                    if len(row) > 1:
                        i = j
                        break
            split_col = []
            for _, w in table[i]:
                if max_w is not None and max_w > 0:
                    split_col.extend([max_w for _ in range(w // max_w)])
                    if w % max_w != 0:
                        split_col.append(w % max_w)
                else:
                    split_col.append(w)

        else:
            # If we don't split the table, this is done by splitting the table into itself
            if max_w is None:
                split_col = [total_cols]
            # If max_w is an int, all tables should be of this width (except maybe the last one)
            elif type(max_w) == int and max_w > 0:
                if max_w > total_cols:
                    split_col = [total_cols]
                else:
                    split_col = [max_w for _ in range(total_cols // max_w)]
                    if total_cols % max_w != 0:
                        split_col.append(total_cols % max_w)
            # If max_w is a list of ints, it defines a custom table split
            elif type(max_w) == list:
                split_col = max_w
            else:
                raise TypeError("'max_w' has to be either None, an int or a list[int].")

        split_tables = [[] for _ in range(len(split_col))]
        for row in table:
            split_rows = split_header_row(row, split_col)
            for i, r in enumerate(split_rows):
                # Only add the rows which are not empty after splitting
                if not all(s == "" for s, _ in r):
                    split_tables[i].append(r)
                    # Each item in split_tables represents one subtable once the original table has been split

        # If decimals is provided, round each score to the amount of decimals set
        if decimals != None and type(decimals) == int:
            for model_scores in scores:
                if highlight_best is not False:
                    if callable(highlight_best):
                        better_than = highlight_best
                    elif highlight_best == "min":
                        better_than = lambda s1, s2 : s1 < s2
                    else:
                        better_than = lambda s1, s2 : s1 > s2
                else:
                    better_than = None
                best_model = None
                best_score = None
                for model, score in model_scores.items():
                    if score is None:
                        model_scores[model] = '--'
                    else:
                        try:
                            model_scores[model] = '{:.{dec}f}'.format(score, dec=decimals)
                            if better_than is not None:
                                if best_score is None or better_than(score, best_score):
                                    best_score = score
                                    best_model = model
                        except (ValueError, TypeError):
                            continue
                if best_model is not None:
                    model_scores[best_model] = "\\textbf{" + model_scores[best_model] + "}"

        scores_per_model = {model: [s.get(model, '--') for s in scores] for model in models}
        
        if sort_models:
            scores_per_model_items = sorted(scores_per_model.items(), key = lambda item : item[0])
        else:
            scores_per_model_items = scores_per_model.items()

        score_strings = [[] for _ in range(len(split_col))]
        i = 0
        for j, w in enumerate(split_col):
            for model, scores in scores_per_model_items:
                if not remove_empty or not all(s=='--' for s in scores[i:i+w]):
                    score_strings[j].append("\\multicolumn{1}{|c|}{" + model + "}" + "\t&" + "\t&".join(str(s) for s in scores[i:i+w]))
            score_strings[j] = "\\\\".join(score_strings[j])
            i += w

        table_string = "\n".join(create_table_string(split_tables[i], score_strings[i]) for i in range(len(split_tables)))

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
        if len(self.evaluation_set) == 0:
            logging.error('Dataset used for evaluating does not contain any examples.')
            raise Exception
        self.usage_encoding = usage_encoding
        self.clustering = clustering

    def evaluate(self, return_labels = False, average = True, save = False, json_path = None, latex_path = None, decimals = None):
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
            encoded_usages = self.usage_encoding.generate_definitions(target_usages, encode_definitions = 'vectors') #TODO: make self.dataset.language optional

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
        scores = self.dataset.evaluate(clustering_results.labels, dataset=self.partition, average=average)

        if save:
            if json_path == None and latex_path == None:
                logging.error("Tried to save results but no path was specified.")
            else:
                model_name = getattr(self.usage_encoding, 'name', type(self.usage_encoding).__name__)

                if hasattr(self.dataset, 'name'):
                    self.save_evaluation_results({'WSI': {self.dataset.name: {metric: {model_name: score} for metric, score in scores.items()}}}, json_path, latex_path, decimals)

                elif hasattr(self.dataset, 'dataset') and self.dataset.dataset != None:
                    parameters = ['dataset', 'language', 'version']
                    dataset_info = {}
                    d = dataset_info
                    for param in parameters:
                        if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                            d[str(getattr(self.dataset, param))] = {}
                            d = d[str(getattr(self.dataset, param))]
                    for metric, score in scores.items():
                        d[metric] = {model_name: score}
                    self.save_evaluation_results({'WSI': dataset_info}, json_path, latex_path, decimals)

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
        if len(self.evaluation_set) == 0:
            logging.error('Dataset used for evaluating does not contain any examples.')
            raise Exception
        self.usage_encoding = usage_encoding

    def evaluate(self, task, label_func = None, save = False, json_path = None, latex_path = None, decimals = None):
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
                encoded_usages = self.usage_encoding.generate_definitions(usage_list, encode_definitions = 'vectors') #TODO: make self.dataset.language optional

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
                        id = pair[f'id{j}']
                    else:
                        id = (pair[f'text{j}'], pair[f'start{j}'], pair[f'end{j}'])
                    embedding_pair.append(encoded_usages[index[id]])

                labels.append(label_func(embedding_pair[0], embedding_pair[1]))

        elif isinstance(self.usage_encoding, PromptModel):
            if task == "graded":
                class WiCGraded(BaseModel):
                    change : float = Field(description='How similar the two occurrences of the word are.',le=1, ge=0)#TODO: perhaps rename change to something else
                self.usage_encoding.structure = WiCGraded #TODO: make sure that the structure is updated!
                for pair in self.evaluation_set:
                    target_usage_list = TargetUsageList([TargetUsage(pair['text1'], [pair['start1'], pair['end1']]),
                                                         TargetUsage(pair['text2'], [pair['start2'], pair['end2']])])
                    labels.append(self.usage_encoding.get_response(target_usage_list, user_prompt_template = 'Please tell me how similar the meaning of the word \'{target}\' is in the following example sentences: \n1. {usage_1}\n2. {usage_2}'))
            elif task == "binary":
                class WiCBinary(BaseModel):
                    change : bool = Field(description='Whether the word has the same meaning or not.')
                self.usage_encoding.structure = WiCBinary #TODO: make sure that the structure is updated!
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
            scores = {'spearman_r': spearman_r.statistic} # Keep rho only

        if save:
            if json_path == None and latex_path == None:
                logging.error("Tried to save results but no path was specified.")
            else:
                model_name = getattr(self.usage_encoding, 'name', type(self.usage_encoding).__name__)
                if hasattr(self.dataset, 'name'):
                    scores_dict = {f'{task.title()} WiC': {self.dataset.name: {metric: {model_name: score} for metric, score in scores.items()}}}
                    self.save_evaluation_results(scores_dict, json_path, latex_path, decimals)

                elif hasattr(self.dataset, 'dataset') and self.dataset.dataset != None:
                    parameters = ['dataset', 'version', 'linguality', 'language']
                    dataset_info = {}
                    d = dataset_info
                    for param in parameters:
                        if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                            d[str(getattr(self.dataset, param))] = {}
                            d = d[str(getattr(self.dataset, param))]
                    for metric, score in scores.items():
                        d[metric] = {model_name: score}
                    scores_dict = {f'{task.title()} WiC': dataset_info}
                    self.save_evaluation_results(scores_dict, json_path, latex_path, decimals)

                else:
                    logging.error("Dataset has no 'name' attribute, nor 'dataset' attribute. Scores could therefore not be saved.")
        
        return scores


class GCDPipeline(Pipeline):
    """
    A pipeline for evaluating the Graded Change Detection (GCD) task.
    Parameters:
        dataset (Union[DWUG,List[Set[TargetUsage]]]): The dataset to evaluate on. Can be a DWUG instance or a list of sets of TargetUsage instances, describing .
        usage_encoding: The model used to encode the usages. Can be a ContextualizedModel, DefinitionGenerator or PromptModel.
        metric (Union[GradedChange,WiDiD]): The metric used to measure the change between usages. Can be a GradedChange (including APD, PRT or PJSD) or WiDiD instance.
        clustering: the clustering algorithm used in the case of PJSD or WiDiD. Needs to be provided for PJSD, defaults to APosterioriaffinityPropagation for WiDiD.
        scores (List): A list of scores to use for evaluation, in the case of loading from TargetUsages.
        dataset_name (str): The name of the dataset, in the case of loading from TargetUsages.
    """
    def __init__(self, dataset : Union[DWUG, List[Set[TargetUsage]]],
                 usage_encoding,
                 metric : Union[GradedChange, WiDiD],
                 clustering = None,
                 scores : List = None,
                 dataset_name : str = None):
        super().__init__()
        if isinstance(dataset, DWUG) or isinstance(dataset, SemEval2020Task1):
            self.dataset = dataset
        else:
            self.dataset = SemanticChangeEvaluationDataset(name=dataset_name)
            self.dataset.load_from_target_usages(dataset, scores)

        self.usage_encoding = usage_encoding
        self.metric = metric
        self.clustering = clustering

    def evaluate(self, save = False, json_path = None, latex_path = None, decimals = None):
        """
            Evaluates on the GCD task. Returns the Spearman correlation between the predicted and ground truth change scores.
            Args:
                save (bool): Whether to save the results to json or LaTeX.
                json_path (str): The path to save the results in JSON format. If None, the results are not saved.
                latex_path (str): The path to save the results in LaTeX format. If None, no LaTeX table is generated.
                decimals (int): The number of decimals to round the scores to in the LaTeX table.
            Returns:
                scores (dict): A dictionary containing the Spearman correlation score (rho).
        """
        change_scores = {}

        if isinstance(self.usage_encoding, PromptModel):
            return None # Is this possible?
        
        elif isinstance(self.usage_encoding, DefinitionGenerator) or isinstance(self.usage_encoding, ContextualizedModel):
            if isinstance(self.dataset, SemEval2020Task1) and self.dataset.dataset not in {"NorDiaChange", "RuShiftEval"}:
                target_usages_t1_all_words = self.dataset.corpus1_lemma.search([target.target for target in self.dataset.graded_task.keys()])
                target_usages_t2_all_words = self.dataset.corpus2_lemma.search([target.target for target in self.dataset.graded_task.keys()])

            for word in self.dataset.target_words:

                if isinstance(self.dataset, DWUG) or (isinstance(self.dataset, SemEval2020Task1) and self.dataset.dataset in {"NorDiaChange", "RuShiftEval"}):
                    target_usages = self.dataset.get_word_usages(word)
                    groupings = set(u.grouping for u in target_usages)
                    try:
                        sorted_groupings = sorted(list(groupings), key = lambda x: int(x.split('-')[0]))
                    except ValueError:
                        sorted_groupings = sorted(list(groupings))
                    target_usages_t1 = [u for u in target_usages if u.grouping == sorted_groupings[0] ]
                    target_usages_t2 = [u for u in target_usages if u.grouping == sorted_groupings[1] ]

                elif isinstance(self.dataset, SemEval2020Task1) and self.dataset.dataset not in {"NorDiaChange", "RuShiftEval"}:
                    target_usages_t1 = target_usages_t1_all_words[word]
                    target_usages_t2 = target_usages_t2_all_words[word]

                elif isinstance(self.dataset, SemanticChangeEvaluationDataset):
                    target_usages_t1 = self.dataset.target_usages_t1[word]
                    target_usages_t2 = self.dataset.target_usages_t2[word]

                if isinstance(self.usage_encoding, DefinitionGenerator):
                    encoded_usages_t1 = self.usage_encoding.generate_definitions(target_usages_t1, encode_definitions='vectors')
                    encoded_usages_t2 = self.usage_encoding.generate_definitions(target_usages_t2, encode_definitions='vectors')

                elif isinstance(self.usage_encoding, ContextualizedModel):
                    encoded_usages_t1 = self.usage_encoding.encode(target_usages_t1)
                    encoded_usages_t2 = self.usage_encoding.encode(target_usages_t2)

                # Measure the change using the metric
                if type(self.metric) == PJSD and self.clustering != None:
                    change = self.metric.compute_scores(encoded_usages_t1, encoded_usages_t2, self.clustering)
                elif type(self.metric) == WiDiD:
                    if self.clustering != None:
                        self.metric == WiDiD(algorithm=self.clustering)
                    _, _, timeseries = self.metric.compute_scores([encoded_usages_t1, encoded_usages_t2])
                    change = timeseries.series[0]
                else:
                    change = self.metric.compute_scores(encoded_usages_t1, encoded_usages_t2)
                change_scores[word] = change

        else:
            logging.error('Model not supported.')
            return None

        spearman_r = self.dataset.evaluate_gcd(change_scores)
        scores = {'spearman_r': spearman_r.statistic} # Keep rho only

        if save:
            if json_path == None and latex_path == None:
                logging.error("Tried to save results but no path was specified.")
            else:
                model_name = getattr(self.usage_encoding, 'name', type(self.usage_encoding).__name__)
                if hasattr(self.dataset, 'name'):
                    scores_dict = {'GCD': {self.dataset.name: {metric: {model_name: score} for metric, score in scores.items()}}}
                    self.save_evaluation_results(scores_dict, json_path, latex_path, decimals)

                elif hasattr(self.dataset, 'dataset'):
                    parameters = ['dataset', 'language', 'version', 'subset']
                    dataset_info = {}
                    d = dataset_info
                    for param in parameters:
                        if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                            d[str(getattr(self.dataset, param))] = {}
                            d = d[str(getattr(self.dataset, param))]
                    for metric, score in scores.items():
                        d[metric] = {model_name: score}
                    scores_dict = {'GCD': dataset_info}
                    self.save_evaluation_results(scores_dict, json_path, latex_path, decimals)

                else:
                    logging.error("Dataset has no 'name' attribute, nor 'version' and 'language' attributes. Scores could therefore not be saved.")     

        return scores
