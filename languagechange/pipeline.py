from typing import List, Set, Union
import math
import json
import logging
import inspect
import os
import re
import csv
import copy
from pydantic import BaseModel, Field
import numpy as np
from languagechange.models.representation.contextualized import ContextualizedModel
from languagechange.models.representation.definition import DefinitionGenerator
from languagechange.models.representation.prompting import PromptModel
from languagechange.usages import TargetUsage, TargetUsageList
from languagechange.models.change.metrics import GradedChange, PJSD
from languagechange.models.change.widid import WiDiD
from languagechange.benchmark import WiC, WSI, SemanticChangeEvaluationDataset, SemEval2020Task1, DWUG

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Utility function to update a dictionary recursively
def deep_update(d, u):
    for k, v in u.items():
        if type(v) == dict:
            if k in d and type(d[k]) == dict:
                d[k] = deep_update(d[k], v.copy())
            else:
                d[k] = v.copy()
        else:
            d[k] = v
    return d

def get_depth(d):
    if type(d) != dict:
        return 0
    return 1 + max([get_depth(v) for v in d.values()])


class WiCBinary(BaseModel):
    wic_label : bool = Field(description='Whether the word has the same meaning or not.')


class WiCGraded(BaseModel):
    wic_label : float = Field(description='How similar the two occurrences of the word are.',le=1, ge=0)


class Pipeline:
    """
        A general class for evaluation pipelines, containing methods common to WSIPipeline, WiCPipeline and
        GCDPipeline, used for saving evaluation results and generating tables from the results.
    """
    def __init__(self):
        pass

    def save_evaluation_results(self, results, json_path : str, table_path : str = None, **kwargs):
        """
            Saves evaluation results to a json file, and optionally generates a table of the results (see 
                self.generate_table). If there is already content in the json file, the results will be merged with 
                this.

            Args:
                results (dict): a nested dictionary containing the new results(s) to add. See for example 
                    WiCPipeline.evaluate for an example usage.
                json_path (str): the path to save results in as a dictionary.
                table_path (Union[str, NoneType], default=None): the path for saving the generated table, see 
                    self.generate table.
        """
        if os.path.exists(json_path):
            with open(json_path, 'r+') as f:
                try:
                    previous_results = json.load(f)
                except json.JSONDecodeError as e:
                    logging.info("Could not save the results to a JSON file due to the following error: ")
                    logging.info(repr(e))
                    return
                results = deep_update(previous_results, results)
                f.seek(0)
                json.dump(results, f, indent=4)
                f.truncate()
                logging.info(f'Evaluation results saved to {json_path}')
        else:
            with open(json_path, 'w') as f:
                try:
                    json.dump(results, f, indent=4)
                except json.JSONDecodeError as e:
                    logging.info("Could not save the results to a JSON file due to the following error: ")
                    logging.info(repr(e))
                    return
                logging.info(f'Evaluation results saved to {json_path}')
        if table_path is not None:
            self.generate_table(results, table_path, **kwargs)
                
    def generate_table(self, 
                            data, 
                            save_path,
                            decimals=None, 
                            remove_headers=0, 
                            max_w=None, 
                            natural_split=False, 
                            remove_empty=False, 
                            sort_models=False, 
                            generate_caption=False, 
                            highlight_best=False, 
                            n_method_cols=1):
        """
            Generates one or more tables of results in LaTeX or TSV format, to be saved in a .tex or .tsv file. Meant 
            to be used together with self.save_evaluation_results.

            Args:
                data (dict): the evaluation results, in a dictionary (similar to that produced by 
                    self.save_evaluation results).
                save_path (str): where to save the .tex or .tsv file.
                decimals (int): the amount of decimals to round evaluation results to.
                remove_headers (int): if >0, the first remove_headers rows of the table are removed.
                max_w (int|List[int]): if not None, split the table into smaller tables. If an int, each table will be 
                    max_w columns wide (excluding the model names to the very left). If a list of ints, the value of 
                    max_w[i] is the width of table i.
                natural_split (int|bool): if True, split the table according to the first row which has a natural 
                    split. If an int >= 0, split the table according to the split of this row.
                remove_empty (bool): If True, remove all rows containing no value.
                sort_models (bool): If True, sort the rows by the names of the models.
                generate_caption (bool): If True, generate a caption for each table.
                highlight_best (bool|Callable|str): If "max", highlight the highest value in each column. If "min", 
                    highlight the lowest value in each column. If a callable, use this as a function to compare values.
                n_method_cols (int): the amount of columns storing method info to the left of the table content, as 
                    opposed to above the table content.
        """

        def get_header_cells_and_scores(data):
            """
                Joins the content of each entry with the width of the cell in the table.
                Merges empty cells together if they belong to the same supercolumn.
                Gets all the methods and their scores in the right order.
            """
            total_depth = get_depth(data)
            header_cells = [[] for _ in range(total_depth-n_method_cols)]
            if header_cells == []:
                scores_per_method = data
                return header_cells, scores_per_method
            
            scores_per_method = []

            def get_rows_rec(data, depth):
                total_w = 0
                empty_space = 0
                for k, v in data.items():
                    if type(v) == dict:
                        # Normal case
                        if get_depth(v) > n_method_cols:
                            # Recursive call to go further down the tree
                            w = get_rows_rec(v, depth + 1)
                            total_w += w
                            header_cells[depth].append((k,w))
                        # The case where we have reached the last row before the model and score, i.e. the row 
                        # describing the metric.
                        else:
                            # Add the metric name to the last row.
                            header_cells[-1].append((k,1))
                            empty_space += 1
                            total_w += 1
                            scores_per_method.append(v)
                if empty_space > 0:
                    for de in range(depth, len(header_cells) - 1):
                        # Add empty space to accommodate for longer columns
                        header_cells[de].extend([('',empty_space)])

                return total_w
            
            get_rows_rec(data, 0)
            return header_cells, scores_per_method
        
        # Gets the table information for method names and scores in the right order.
        def get_content_cells(methods, scores):
            content_cells = []

            def get_content_cells_rec(m, scores, row, col):
                if not type(m) == dict:
                    while len(content_cells) < row + 1:
                        content_cells.append([None for _ in range(n_method_cols+n_content_cols)])
                    for c, s in enumerate(scores):
                        content_cells[row][col+c] = (s, 1)
                    return 1
                n_leaves = 0

                # Sort models alphabetically if we have reached the {model: score} dict
                if sort_models and all(type(v) != dict for v in m.values()):
                    items = sorted(m.items(), key = lambda i : i[0])
                else:
                    items = m.items()

                for k, v in items:
                    if type(v) != dict:
                        c = n_method_cols - 1
                    else:
                        c = col
                    l = get_content_cells_rec(v,[s.get(k, None) if s is not None else None for s in scores], row,c+1)
                    content_cells[row][c] = (k, l)
                    n_leaves += l
                    row += l
                return n_leaves
            
            get_content_cells_rec(methods, scores, 0, 0)
            return content_cells

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
        
        def split_content_and_side_cells(content_cells, side_cells, split_cols):
            """
                Splits the content and side parts of a table into multiple tables.
            """
            split_contents = []
            split_sides = []
            side_width = len(side_cells[0])
            i = 0
            for w in split_cols:
                side_cells_to_add = []
                content_cells_to_add = []
                row_i = 0
                # Take the current content along with the side cells, which are the same for all split tables
                for content, side in zip([c[i:i+w] for c in content_cells], side_cells):
                    if remove_empty and all (c is None for c, _ in content):
                        # If the row is empty and not the first one, decrease the multirow height of rows above
                        if row_i > 0:
                            for c in range(side_width):
                                r = row_i-1
                                while side_cells_to_add[r][c] is None and r > 0:
                                    r -= 1
                                if side_cells_to_add[r][c] is not None and side_cells_to_add[r][c][1] > 1:
                                    side_cells_to_add[r][c] = (side_cells_to_add[r][c][0], side_cells_to_add[r][c][1] - 1)
                    # If the row is not empty, add it
                    else:
                        side_cells_to_add.append(side.copy())
                        content_cells_to_add.append(content)
                        row_i += 1
                split_contents.append(content_cells_to_add)
                split_sides.append(side_cells_to_add)
                i += w
            return split_contents, split_sides
        
        def get_horizontal_lines(header_cells):
            """
                Draws horizontal lines between table rows where it fits.
            """
            line_strings = ["" for _ in range(len(header_cells))]
            for i, r in enumerate(header_cells[:-2]):
                index1 = 0
                for s1, w1 in r:
                    index2 = 0
                    match = False
                    for s2, w2 in header_cells[i+1]:
                        # If the two rows have matching multicolumns and one of them is empty, don't draw a horizontal 
                        # line between them
                        if index2 == index1 and index2 + w2 == index1 + w1 and (s1 == '' or s2 == ''):
                            match = True
                        index2 += w2
                    if not match:
                        line_strings[i] += "\\cline{" + str(index1 + n_method_cols + 1) + "-" + str(index1 + w1 + n_method_cols) + "}"
                    index1 += w1
            # Before the metrics, add a complete horizontal line
            if len(line_strings) >= 2:
                line_strings[-2] = "\\cline{"+str(n_method_cols+1)+"-" + str(sum(w for _, w in header_cells[0])+(n_method_cols)) + "}"
            return line_strings
        
        def render_header_row(row):
            return f"\\multicolumn{{{n_method_cols}}}{{c}}{{}}\t&" + "\t&".join([f"\\multicolumn{{{w}}}{{|c|}}{{{s}}}" for (s,w) in row])

        def render_header_row_tsv(row):
            r = [""] * n_method_cols
            for (s,w) in row:
                r.extend([s] + [""] * (w - 1))
            return r
        
        def format_scores(d):
            """
                Rounds scores to a number of decimals, if provided, and optionally sorts the score rows by model name.
            """
            if not all (type(v) == dict for v in d.values()):
                best_model = None
                best_score = None
                model_scores = d
                for model, score in model_scores.items():
                    if score is None:
                        model_scores[model] = '--'
                    else:
                        try:
                            # If decimals is provided, round each score to the amount of decimals set
                            if decimals != None and type(decimals) == int:
                                model_scores[model] = '{:.{dec}f}'.format(score, dec=decimals)
                            else:
                                model_scores[model] = str(score)
                            if better_than is not None:
                                if best_score is None or better_than(score, best_score):
                                    best_score = score
                                    best_model = model
                        except (ValueError, TypeError):
                            continue
                if best_model is not None and save_format == "tex":
                    model_scores[best_model] = ("\\textbf{" + model_scores[best_model] + "}")
            else:
                for v in d.values():
                    format_scores(v)
        
        def render_content_rows(side_rows, content_rows, n_content_cols):
            score_string = []

            def format_side_cell(c):
                if c is None:
                    return ""
                if c[1] > 1:
                    return "\\multirow{" + str(c[1]) + "}{*}{" + str(c[0]) + "}"
                return str(c[0])

            for r, (side_row, content_row) in enumerate(zip(side_rows, content_rows)):
                side_cells = list(map(format_side_cell, side_row))
                content_cells = [c[0] if c[0] is not None else "--" for c in content_row]
                if r == 0:
                    lines = ["\\hline"]
                else:
                    lines =  ["\\cline{"+str(i+1)+"-"+str(n_method_cols+n_content_cols)+"}" if c != "" else "" 
                                for i, c in enumerate(side_cells[:-1])]
                row_string = "".join(lines) + "\t" + "\t&".join(side_cells + content_cells)
                score_string.append(row_string)
            return "\\\\\n".join(score_string)

        def render_content_rows_tsv(side_rows, content_rows):
            score_string = []

            def format_side_cell(c):
                if c is None:
                    return ""
                return str(c[0])

            for side_row, content_row in zip(side_rows, content_rows):
                side_cells = list(map(format_side_cell, side_row))
                content_cells = [c[0] if c[0] is not None else "--" for c in content_row]
                row_string = side_cells + content_cells
                score_string.append(row_string)
            return score_string
            
        # Puts together the different parts of a table
        def create_table_string(header_rows, side_rows, content_rows, n_content_cols):
            columns_str = "|"+"|".join(["c"] * (n_method_cols+n_content_cols))+"|"

            table_beginning = """
\\begin{table}[h]
    \\centering
    \\begin{tabular}{"""+ columns_str +"}\\cline{"+str(n_method_cols+1)+"-"+str(n_content_cols+n_method_cols)+"}"
                
            table_end = """
        \hline
    \\end{tabular}""" + (("\n\\caption{Evaluation results on the " + header_rows[0][0][0] + " task.}") if generate_caption else "") + """
\\end{table}"""
            
            line_strings = get_horizontal_lines(header_rows)
            header_string = "".join(render_header_row(row) + "\\\\\n" + line_strings[i] for i, row in enumerate(header_rows))
            score_string = render_content_rows(side_rows, content_rows, n_content_cols)            

            table_string = table_beginning + header_string + score_string + "\\\\\n" + table_end
            table_string = re.sub("_", "\_", table_string)

            return table_string

        def create_table_string_tsv(header_rows, side_rows, content_rows):
            header_string = list(map(render_header_row_tsv, header_rows))
            score_string = render_content_rows_tsv(side_rows, content_rows)
            return header_string + score_string

        if save_path.endswith(".tex"):
            save_format = "tex"
        elif save_path.endswith(".tsv"):
            save_format = "tsv"
        else:
            logging.error("save_path needs to end in .tex or .tsv")
            raise ValueError
    
        data = copy.deepcopy(data)
        header_cells, scores_per_method = get_header_cells_and_scores(data)
        header_cells = header_cells[remove_headers:]
        n_content_cols = sum(w for _, w in header_cells[0])

        # Split the table according to a natural subdivision in the data
        if natural_split:
            if header_cells == []:
                logging.error("The table has to have headers to split it naturally.")
                raise ValueError
            # If an int, use the natural split of this row
            if isinstance(natural_split, int):
                i = natural_split
            # Otherwise, choose the first row that has a split
            else:
                i = next((j for j, row in enumerate(header_cells) if len(row) > 1), 0)
            if max_w is not None and max_w > 0:
                split_cols = []
                for _, w in header_cells[i]:
                    split_cols.extend([max_w for _ in range(w // max_w)])
                    if w % max_w != 0:
                        split_cols.append(w % max_w)
            else:
                split_cols = [w for _, w in header_cells[i]]

        else:
            # If we don't split the table, this is done by splitting the table into itself
            if max_w is None:
                split_cols = [n_content_cols]
            # If max_w is an int, all tables should be of this width (except maybe the last one)
            elif type(max_w) == int and max_w > 0:
                if max_w > n_content_cols:
                    split_cols = [n_content_cols]
                else:
                    split_cols = [max_w for _ in range(n_content_cols // max_w)]
                    if n_content_cols % max_w != 0:
                        split_cols.append(n_content_cols % max_w)
            # If max_w is a list of ints, it defines a custom table split
            elif type(max_w) == list:
                split_cols = max_w
            else:
                raise TypeError("'max_w' has to be either None, an int > 0 or a list[int].")

        split_header_rows = [[] for _ in range(len(split_cols))]

        for row in header_cells:
            split_rows = split_header_row(row, split_cols)
            for i, r in enumerate(split_rows):
                # Only add the rows which are not empty after splitting
                if not all(s == "" for s, _ in r):
                    split_header_rows[i].append(r)
                    # Each item in split_tables represents one subtable once the original table has been split
        
        if highlight_best is not False:
            if callable(highlight_best):
                better_than = highlight_best
            elif highlight_best == "min":
                better_than = lambda s1, s2 : s1 < s2
            else:
                better_than = lambda s1, s2 : s1 > s2
        else:
            better_than = None

        all_methods = dict()
        for scores in scores_per_method:
            format_scores(scores)
            all_methods = deep_update(all_methods, scores)
            
        content_cells = get_content_cells(all_methods, scores_per_method)

        split_content_rows, split_side_rows = split_content_and_side_cells([c[n_method_cols:] for c in content_cells], [c[:n_method_cols] for c in content_cells], split_cols)

        if save_format == "tex":
            table_string = "\n".join([create_table_string(split_header_rows[i], split_side_rows[i], split_content_rows[i], split_cols[i]) for i, _ in enumerate(split_cols)])

            # Save the LaTeX string to a .tex file
            if save_path.endswith(".tex"):
                with open(save_path,'w+') as f:
                    f.write(table_string)
            else:
                raise Exception("The file needs to end in .tex")
        elif save_format == "tsv":
            with open(save_path, 'w', newline = "") as f:
                writer = csv.writer(f, delimiter="\t")
                for i, _ in enumerate(split_cols):
                    tsv_content = create_table_string_tsv(split_header_rows[i], split_side_rows[i], split_content_rows[i])
                    writer.writerows(tsv_content + [""])
        logging.info(f"Wrote results to {save_path}.")


class WSIPipeline(Pipeline):
    """
        Pipeline for evaluating the Word Sense Induction (WSI) task.

        This pipeline:
        1. encodes usages using either a ContextualizedModel or a definition generator 
        (`DefinitionGenerator`) producing embeddings
        2. clusters the resulting embeddings using the provided clustering algorithm
        3. evaluates cluster assignments with ARI and purity

        Parameters:
            dataset (Union[WSI,List[TargetUsage],TargetUsageList]): The dataset to evaluate on. Can
            be a WSI instance or a list of TargetUsage instances, describing WSI examples.
            usage_encoding: The model used to encode the usages. Can be a ContextualizedModel or a 
            DefinitionGenerator.
            clustering (Clustering): A Clustering object (see 
            languagechange.models.meaning.clustering) containing a get_cluster_results method, to 
            use to cluster the encoded usages.
            partition (str, default="test"): Dataset split to evaluate on: commonly "train", "dev", 
            or "test".
            shuffle (bool, default=True): Whether to shuffle when splitting the dataset if it is 
            loaded from raw usages.
            labels (List): A list of labels to use for evaluation, in the case of loading from 
            TargetUsages.
            dataset_name (str): The name of the dataset, in the case of loading from TargetUsages.
    """
    def __init__(self, dataset, usage_encoding, clustering, partition = 'test', shuffle = True, labels=[], dataset_name = None):
        super().__init__()
        if not (isinstance(usage_encoding, ContextualizedModel) or isinstance(usage_encoding, DefinitionGenerator)):
            logging.error("usage_encoding must be either a ContextualizedModel or a DefinitionGenerator.")
            raise TypeError
        self.usage_encoding = usage_encoding

        if isinstance(dataset, WSI):
            self.dataset = dataset
        else:
            self.dataset = WSI(name=dataset_name)
            self.dataset.load_from_target_usages(dataset, labels)
            self.dataset.split_train_dev_test(shuffle=shuffle)
        self.partition = partition
        self.evaluation_set = self.dataset.get_dataset(self.partition)
        if len(self.evaluation_set) == 0:
            logging.error('Dataset used for evaluating does not contain any examples.')
            raise Exception
        
        self.clustering = clustering

    def evaluate(self, return_labels = False, average = True, json_path = None, table_path = None, **kwargs):
        """
            Evaluate on the WSI task.

            Args:
                return_labels (bool, default=False): if True, return not only the scores but also the cluster labels.
                average (bool, default=True): whether to average across words for ARI and purity.
                json_path (Union[str, NoneType], default=None): if a file path (.json) is specified, try to save the 
                    results to this json file.
                table_path (Union[str, NoneType], default=None): if a file path (.tex or .tsv). is specified and 
                    json_path is also specified, add the results to a table in this file.

            Returns:
                scores (dict): a dictionary containing the ARI and putiry scores.
        """
        target_usages = TargetUsageList()

        for example in self.evaluation_set:
            if 'id' in example:
                target_usages.append(TargetUsage(example['text'], [example['start'],example['end']], id=example['id']))
            else:
                target_usages.append(TargetUsage(example['text'], [example['start'],example['end']]))

        if isinstance(self.usage_encoding, DefinitionGenerator):
            encoded_usages = self.usage_encoding.generate_definitions(target_usages, encode_definitions = 'vectors') #TODO: make self.dataset.language optional
        
        elif isinstance(self.usage_encoding, ContextualizedModel):
            encoded_usages = self.usage_encoding.encode(target_usages)

        # Cluster the encoded usages
        clustering_results = self.clustering.get_cluster_results(encoded_usages)

        # Compute ARI and purity scores
        scores = self.dataset.evaluate(clustering_results.labels, dataset=self.partition, average=average)

        if json_path is not None:
            model_name = getattr(self.usage_encoding, 'name', type(self.usage_encoding).__name__)

            if hasattr(self.dataset, 'name'):
                self.save_evaluation_results({'WSI': {self.dataset.name: {metric: {model_name: score} for metric, score in scores.items()}}}, json_path, table_path=table_path, **kwargs)

            elif hasattr(self.dataset, 'dataset') and self.dataset.dataset is not None:
                parameters = ['dataset', 'language', 'version', 'subset']
                dataset_info = {}
                d = dataset_info
                for param in parameters:
                    if hasattr(self.dataset, param) and getattr(self.dataset, param) is not None:
                        d[str(getattr(self.dataset, param))] = {}
                        d = d[str(getattr(self.dataset, param))]
                for metric, score in scores.items():
                    d[metric] = {model_name: score}
                self.save_evaluation_results({'WSI': dataset_info}, json_path, table_path=table_path, **kwargs)

            else:
                logging.error("Dataset has no 'name' attribute, nor 'dataset' attribute. Scores could therefore not be saved.")

        if return_labels:
            return scores, clustering_results.labels
        return scores

        
class WiCPipeline(Pipeline):
    """
    A pipeline for evaluating the Word-in-Context (WiC) task.

    This pipeline:
        1. loads a dataset that can be used for the WiC task
        2. either
            a) encodes usages of the dataset using either a ContextualizedModel or a DefinitionGenerator producing 
            embeddings, and then compute similarities within embedding pairs corresponding to the WiC examples, or
            b) uses a PromptModel to directly make pairwise similarity judgments
        3. evaluates the similarity judgments using accuracy and F1 or Spearman correlation, depending on the kind
        of task.

    Parameters:
        dataset (Union[DWUG,List[Set[TargetUsage]]]): The dataset to evaluate on. Can be a DWUG instance or a list of 
            sets of TargetUsage instances, describing .
        usage_encoding: The model used to encode the usages. Can be a ContextualizedModel, DefinitionGenerator or 
            PromptModel.
        metric (Union[GradedChange,WiDiD]): The metric used to measure the change between usages. Can be a 
            GradedChange (including APD, PRT or PJSD) or WiDiD instance.
        clustering: the clustering algorithm used in the case of PJSD or WiDiD. Needs to be provided for PJSD, 
            defaults to APosterioriaffinityPropagation for WiDiD.
        scores (List): A list of scores to use for evaluation, in the case of loading from TargetUsages.
        dataset_name (str): The name of the dataset, in the case of loading from TargetUsages.
    """
    def __init__(self, dataset, usage_encoding, partition = 'test', shuffle = True, labels=[], dataset_name = None):
        super().__init__()
        if not (isinstance(usage_encoding, ContextualizedModel) or isinstance(usage_encoding, DefinitionGenerator) or isinstance(usage_encoding, PromptModel)):
            logging.error("usage_encoding must be either a ContextualizedModel, a DefinitionGenerator or a PromptModel.")
            raise TypeError
        self.usage_encoding = usage_encoding
        if isinstance(dataset, WiC):
            self.dataset = dataset
        else:
            self.dataset = WiC(name=dataset_name)
            self.dataset.load_from_target_usages(dataset, labels)
            self.dataset.split_train_dev_test(shuffle=shuffle)

        self.partition = partition
        self.evaluation_set = self.dataset.get_dataset(self.partition)
        if len(self.evaluation_set) == 0:
            logging.error('Dataset used for evaluating does not contain any examples.')
            raise ValueError

    def evaluate(self, task, label_func = None, json_path = None, table_path = None, **kwargs):
        """
            Evaluates on the WiC task. Returns accuracy and f1 scores if task='binary', Spearman correlation if 
            task='graded'.

            Args:
                task (str): the kind of Word-in-Context task ('binary' or 'graded')
                label_func (Callable, default=None): an optional function to use for pairwise similarity judgments of
                    embeddings. By default, cosine similarity is used if task='graded', and a binary threshold at 0.5
                    if task='binary'.
                json_path (Union[str, NoneType], default=None): if a file path (.json) is specified, try to save the 
                    results to this json file.
                table_path (Union[str, NoneType], default=None): if a file path (.tex or .tsv). is specified and 
                    json_path is also specified, add the results to a table in this file.

            Returns:
                scores (dict): a dictionary of scores (accuracy and f1 or Spearman correlation)
        """
        if task not in {'binary','graded'}:
            logging.error(f"Invalid argument for 'task', should be one of ['binary', 'graded']")
            raise ValueError
        
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

            if label_func is None:
                if task == "graded":
                    label_func = lambda e1, e2 : np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2))
                else:
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
                template = 'Please tell me how similar the meaning of the word \'{target}\' is in the following example sentences: \n1. {usage_1}\n2. {usage_2}'
                self.usage_encoding.set_structure(WiCGraded)
            else:
                template = 'Please tell me if the meaning of the word \'{target}\' is the same in the following example sentences: \n1. {usage_1}\n2. {usage_2}'
                self.usage_encoding.set_structure(WiCBinary)
                
            for pair in self.evaluation_set:
                target_usage_list = TargetUsageList([TargetUsage(pair['text1'], [pair['start1'], pair['end1']]),
                                                        TargetUsage(pair['text2'], [pair['start2'], pair['end2']])])
                label = int(self.usage_encoding.get_response(target_usage_list, user_prompt_template=template, response_attribute="wic_label"))
                labels.append(label)
        
        if task == 'binary':
            acc = self.dataset.evaluate_accuracy(labels, self.partition)
            f1 = self.dataset.evaluate_f1(labels, self.partition)
            scores = {'accuracy': acc, 'f1': f1}

        elif task == 'graded':
            spearman_r = self.dataset.evaluate_spearman(labels, self.partition)
            scores = {'spearman_r': None if math.isnan(spearman_r.statistic) else spearman_r.statistic} # Keep rho only

        if json_path is not None:
            model_name = getattr(self.usage_encoding, 'name', type(self.usage_encoding).__name__)
            if hasattr(self.dataset, 'name'):
                scores_dict = {f'{task.title()} WiC': {self.dataset.name: {metric: {model_name: score} for metric, score in scores.items()}}}
                self.save_evaluation_results(scores_dict, json_path, table_path=table_path, **kwargs)

            elif hasattr(self.dataset, 'dataset') and self.dataset.dataset != None:
                parameters = ['dataset', 'language', 'version', 'linguality', 'subset']
                dataset_info = {}
                d = dataset_info
                for param in parameters:
                    if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                        d[str(getattr(self.dataset, param))] = {}
                        d = d[str(getattr(self.dataset, param))]
                for metric, score in scores.items():
                    d[metric] = {model_name: score}
                scores_dict = {f'{task.title()} WiC': dataset_info}
                self.save_evaluation_results(scores_dict, json_path, table_path=table_path, **kwargs)

            else:
                logging.error("Dataset has no 'name' attribute, nor 'dataset' attribute. Scores could therefore not be saved.")
        
        return scores


class GCDPipeline(Pipeline):
    """
    A pipeline for evaluating the Graded Change Detection (GCD) task.

    This pipeline:
        1. loads a dataset that can be used for GCD and extracts the usages from two time periods
        2. encodes usages (optionally sampling n usages) using either a ContextualizedModel or a DefinitionGenerator 
        producing embeddings
        3. computes the change scores for each word using one of the standard metrics (APD, PRT, PJSD, WiDiD).
        4. evaluates the change scores using Spearman correlation

    Parameters:
        dataset (Union[DWUG,List[Set[TargetUsage]]]): The dataset to evaluate on. Can be a DWUG instance or a list of 
            sets of TargetUsage instances, describing .
        usage_encoding: The model used to encode the usages. Can be a ContextualizedModel, DefinitionGenerator or 
            PromptModel.
        metric (Union[GradedChange,WiDiD]): The metric used to measure the change between usages. Can be a 
            GradedChange (including APD, PRT or PJSD) or WiDiD instance.
        clustering: the clustering algorithm used in the case of PJSD or WiDiD. Needs to be provided for PJSD, 
            defaults to APosterioriaffinityPropagation for WiDiD.
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

    def evaluate(self, n_sampled_usages = 0, random_seed = None, json_path = None, table_path = None, **kwargs):
        """
            Evaluates on the GCD task. Returns the Spearman correlation between the predicted and ground truth change 
            scores.

            Args:
                n_sampled_usages (int): the amount of usages to sample for each target word and time period. If 0, use 
                    all usages.
                random_seed (int): the seed for numpy.random.default_rng, used when sampling usages. If None, no seed 
                    is used.
                json_path (Union[str, NoneType], default=None): if a file path (.json) is specified, try to save the 
                    results to this json file.
                table_path (Union[str, NoneType], default=None): if a file path (.tex or .tsv). is specified and 
                    json_path is also specified, add the results to a table in this file.

            Returns:
                scores (dict): A dictionary containing the Spearman correlation score (rho).
        """
        change_scores = {}

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

            def get_sampled_usages(target_usages, generator, n_samples):
                if n_samples < len(target_usages):
                    return generator.choice(target_usages, size=n_samples, replace=False).tolist()
                return target_usages

            if n_sampled_usages > 0:
                rng = np.random.default_rng(seed=random_seed)
                target_usages_t1 = get_sampled_usages(target_usages_t1, rng, n_sampled_usages)
                target_usages_t2 = get_sampled_usages(target_usages_t2, rng, n_sampled_usages)

            if isinstance(self.usage_encoding, DefinitionGenerator):
                encoded_usages_t1 = self.usage_encoding.generate_definitions(target_usages_t1, encode_definitions='vectors')
                encoded_usages_t2 = self.usage_encoding.generate_definitions(target_usages_t2, encode_definitions='vectors')

            elif isinstance(self.usage_encoding, ContextualizedModel):
                encoded_usages_t1 = self.usage_encoding.encode(target_usages_t1)
                encoded_usages_t2 = self.usage_encoding.encode(target_usages_t2)

            # Measure the change using the metric
            if isinstance(self.metric, PJSD) and self.clustering is not None:
                change = self.metric.compute_scores(encoded_usages_t1, encoded_usages_t2, self.clustering)
            elif isinstance(self.metric, WiDiD):
                if self.clustering is not None:
                    self.metric = WiDiD(algorithm=self.clustering)
                _, _, timeseries = self.metric.compute_scores([encoded_usages_t1, encoded_usages_t2])
                change = timeseries.series[0]
            else:
                change = self.metric.compute_scores(encoded_usages_t1, encoded_usages_t2)
            change_scores[word] = change

        spearman_r = self.dataset.evaluate_gcd(change_scores)
        scores = {'spearman_r': None if math.isnan(spearman_r.statistic) else spearman_r.statistic} # Keep rho only

        if json_path is not None:
            model_name = getattr(self.usage_encoding, 'name', type(self.usage_encoding).__name__)
            if hasattr(self.dataset, 'name'):
                scores_dict = {'GCD': {self.dataset.name: {metric: {type(self.metric).__name__: {model_name: score}} for metric, score in scores.items()}}}
                self.save_evaluation_results(scores_dict, json_path, table_path=table_path, **kwargs)

            elif hasattr(self.dataset, 'dataset'):
                parameters = ['dataset', 'language', 'version', 'subset']
                dataset_info = {}
                d = dataset_info
                for param in parameters:
                    if hasattr(self.dataset, param) and getattr(self.dataset, param) != None:
                        d[str(getattr(self.dataset, param))] = {}
                        d = d[str(getattr(self.dataset, param))]
                for metric, score in scores.items():
                    d[metric] = {type(self.metric).__name__: {model_name: score}}
                scores_dict = {'GCD': dataset_info}
                self.save_evaluation_results(scores_dict, json_path, table_path=table_path, **kwargs)

            else:
                logging.error("Dataset has no 'name' attribute, nor 'version' and 'language' attributes. Scores could therefore not be saved.")     

        return scores
