from languagechange.models.representation.contextualized import ContextualizedModel
from languagechange.models.representation.definition import ChatModelDefinitionGenerator
from languagechange.models.representation.prompting import PromptModel
from languagechange.models.meaning.clustering import Clustering
from languagechange.usages import TargetUsage, TargetUsageList
from languagechange.models.change.metrics import GradedChange
from languagechange.benchmark import WiC, WSD, WSI, SemEval2020Task1, DWUG
from pydantic import BaseModel, Field
from typing import List, Set, Union
import numpy as np
import logging
import inspect

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Pipeline:
    def __init__(self):
        pass

    def generate_latex_table(self, scores_per_model,
                   path,
                   scores_to_compute = {"WSD": ["Accuracy","F1"], "WSI":["ARI","Purity"], "WiC":["Accuracy","F1","Spearman"], "GCD":["Spearman"]}):
        metrics_per_task = [len(t) for t in scores_to_compute.values()]
        columns_str = "|"+"|".join(["c"] + ["c"*t for t in metrics_per_task])+"|"

        score_names = []
        for s in scores_to_compute.values():
            score_names.extend(s)

        table_beginning = """\\begin{table}[h]
        \\centering
        \\begin{tabular}{"""+ columns_str +"""}
            \hline
            model\t& """ + "\t&".join("\\multicolumn{"+str(len(s))+"}{|c|}{"+t+"}" for t, s in scores_to_compute.items())+"\\\\\n"+ "\t&" + "\t&".join(score_names) + "\\\\ \\hline\n"
        
        table_end = """
            \hline
        \\end{tabular}
    \\end{table}
        """

        table_data = ""
        table_scores = {}
        for model in scores_per_model:
            model_scores = scores_per_model[model]
            for task in scores_to_compute.keys():
                table_scores[task] = {}
                if task not in model_scores:
                    for metric in scores_to_compute[task]:
                        table_scores[task][metric] = "--"
                else:
                    for metric in scores_to_compute[task]:
                        if metric not in model_scores[task] or model_scores[task][metric] == None:
                            table_scores[task][metric] = "--"
                        else:
                            table_scores[task][metric] = str(model_scores[task][metric])
            scores_list = []
            for t in table_scores.values():
                scores_list.extend(t.values())
            table_data += model.replace('_', '\_') + "\t&" + "\t&".join(scores_list) + " \\\\\n"
        
        latex_string = table_beginning + table_data + table_end

        with open(path,'w+') as f:
            f.write(latex_string)


class WSIPipeline(Pipeline):
    def __init__(self, dataset, usage_encoding, clustering, partition = 'test', shuffle = True, labels=[]):
        super().__init__()
        if isinstance(dataset, WSI):
            self.dataset = dataset
        else:
            self.dataset = WSI() #TODO: exceptions
            self.dataset.load_from_target_usages(dataset, labels)
            self.dataset.split_train_dev_test(shuffle=shuffle)

        self.partition = partition
        self.evaluation_set = self.dataset.get_dataset(self.partition)
        self.usage_encoding = usage_encoding
        self.clustering = clustering

    def evaluate(self, return_labels = False):
        """
            Evaluate on the WSI task. Returns the ARI and purity scores and optionally the clustering labels.
        """
        target_usages = TargetUsageList()

        for example in self.evaluation_set:
            if 'id' in example:
                target_usages.append(TargetUsage(example['text'], [example['start'],example['end']], id=example['id']))
            else:
                target_usages.append(TargetUsage(example['text'], [example['start'],example['end']]))

        if isinstance(self.usage_encoding, ChatModelDefinitionGenerator): #TODO: make a more general definition generation class and add this as the instance type here
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

        if return_labels:
            return scores, clustering_results.labels
        return scores

        
class WiCPipeline(Pipeline):
    def __init__(self, dataset, usage_encoding, partition = 'test', shuffle = True, labels=[]):
        super().__init__()
        if isinstance(dataset, WiC):
            self.dataset = dataset
        else:
            self.dataset = WiC() #TODO: exceptions
            self.dataset.load_from_target_usages(dataset, labels)
            self.dataset.split_train_dev_test(shuffle=shuffle)

        self.partition = partition
        self.evaluation_set = self.dataset.get_dataset(self.partition)
        self.usage_encoding = usage_encoding

    def evaluate(self, task, label_func = None):
        """
            Evaluates on the WiC task. Returns accuracy and f1 scores if task='binary', Spearman correlation if task='graded'.
        """
        if task not in {'binary','graded'}:
            logging.error(f'Invalid argument for \'task\', should be one of [\'binary\', \'graded\']')
            return None
        
        labels = []

        if isinstance(self.usage_encoding, ChatModelDefinitionGenerator) or isinstance(self.usage_encoding, ContextualizedModel): #TODO: make a more general definition generation class and add this here
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

            elif isinstance(self.usage_encoding, ChatModelDefinitionGenerator):
                encoded_usages = self.usage_encoding.generate_definitions(usage_list, encode_definitions = 'vectors')

            if label_func == None:
                if task == "graded":
                    label_func = lambda e1, e2 : np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2))
                elif task == "binary":
                    label_func = lambda e1, e2 : int(np.dot(e1, e2)/(np.linalg.norm(e1) * np.linalg.norm(e2)) > 0.5)
            
            # else: check if callable
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
                    change : float = Field(description='How similar the two occurrences of the word are.',le=1, ge=0) #perhaps rename change to something else
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
            return acc, f1
        
        elif task == 'graded':
            spearman_r = self.dataset.evaluate_spearman(labels, self.partition)
            return spearman_r


class GCDPipeline(Pipeline):
    def __init__(self, dataset : Union[DWUG, List[Set[TargetUsage]]],
                 usage_encoding,
                 metric : GradedChange):
        super().__init__()
        if isinstance(dataset, DWUG):
            self.dataset = dataset
        else:
            pass
        self.usage_encoding = usage_encoding
        self.metric = metric

    def evaluate(self):
        """
            Evaluates on the GCD task. Returns the Spearman correlation between the predicted and ground truth change scores.
        """
        change_scores = {}

        if isinstance(self.usage_encoding, PromptModel):
            return None # Is this possible?
        
        elif isinstance(self.usage_encoding, ChatModelDefinitionGenerator) or isinstance(self.usage_encoding, ContextualizedModel):
            for target in self.dataset.graded_task.keys():
                word = target.target
                target_usages = self.dataset.get_word_usages(word)
                target_usages_t1 = [u for u in target_usages if u.grouping == "1" ]
                target_usages_t2 = [u for u in target_usages if u.grouping == "2" ]

                if isinstance(self.usage_encoding, ChatModelDefinitionGenerator):
                    encoded_usages_t1 = self.usage_encoding.generate_definitions(target_usages_t1[:1], encode_definitions='vectors')
                    encoded_usages_t2 = self.usage_encoding.generate_definitions(target_usages_t2[:1], encode_definitions='vectors')

                elif isinstance(self.usage_encoding, ContextualizedModel):
                    encoded_usages_t1 = self.usage_encoding.encode(target_usages_t1[:1])
                    encoded_usages_t2 = self.usage_encoding.encode(target_usages_t2[:1]) #change!

                # Measure the change using the metric
                change = self.metric.compute_scores(encoded_usages_t1, encoded_usages_t2)

                change_scores[word] = change

        else:
            logging.error('Model not supported.')
            return None

        spearman_score = self.dataset.evaluate_gcd(change_scores)

        return spearman_score
