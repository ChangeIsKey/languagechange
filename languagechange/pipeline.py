from languagechange.models.representation.contextualized import ContextualizedModel
from languagechange.models.representation.definition import ChatModelDefinitionGenerator
from languagechange.models.representation.prompting import PromptModel
from languagechange.models.meaning.clustering import Clustering
from languagechange.usages import TargetUsage, TargetUsageList
from languagechange.models.change.metrics import GradedChange
from pydantic import BaseModel, Field
from typing import List, Set
from scipy.spatial.distance import cdist, cosine

class Pipeline:
    def __init__(self):
        pass


class WSIPipeline(Pipeline):
    def __init__(self):
        super().__init__()

    def get_clusters(self, target_usages : List[TargetUsage] | TargetUsageList, 
                     usage_encoding, 
                     clustering : Clustering):
        if isinstance(usage_encoding, ChatModelDefinitionGenerator): #TODO: make a more general definition generation class and add this here
            encoded_usages = usage_encoding.generate_definitions(target_usages, encode = True)
        elif isinstance(usage_encoding, PromptModel):
            pass # Is this possible?
        elif isinstance(usage_encoding, ContextualizedModel):
            encoded_usages = usage_encoding.encode(target_usages)

        # Cluster the encoded usages
        clustering_results = clustering.get_cluster_results(encoded_usages)

        return clustering_results.labels

        
class WiCPipeline(Pipeline):
    def __init__(self):
        super().__init__()

    def get_labels(target_usage_pairs : List[List[TargetUsage]], 
                   usage_encoding, 
                   task,
                   label_func = None):
        scores = []

        if isinstance(usage_encoding, ChatModelDefinitionGenerator) or isinstance(usage_encoding, ContextualizedModel): #TODO: make a more general definition generation class and add this here
            # Find the unique usages among all pairs
            index = dict() # Index to point to the right position in the usage/embeddings list when comparing usages in pairs TODO: test the index
            i = 0
            usage_list = TargetUsageList()
            for pair in target_usage_pairs:
                for p in pair:
                    if (p.text(), p.offsets) not in index:
                        index[(p.text(), p.offsets)] = i
                        i += 1
                        usage_list.append(p)
            if isinstance(usage_encoding, ContextualizedModel):
                encoded_usages = usage_encoding.encode(usage_list)
            elif isinstance(usage_encoding, ChatModelDefinitionGenerator):
                encoded_usages = usage_encoding.generate_definitions(usage_list, encode = True)

            if label_func == None:
                if task == "graded":
                    label_func = cosine
                if task == "binary":
                    label_func = lambda e1, e2 : int(cosine(e1, e2) > 0.5)
            
            # else: check if callable
            for pair in target_usage_pairs:
                e1 = encoded_usages[index[pair[0]]]
                e2 = encoded_usages[index[pair[1]]]
                scores.append(label_func(e1, e2))

        elif isinstance(usage_encoding, PromptModel):
            if task == "graded":
                class WiCGraded(BaseModel):
                    change : float = Field(description='How similar the two occurrences of the word are.',le=1, ge=0) #perhaps rename change to something else
                usage_encoding.structure = WiCGraded
                for pair in target_usage_pairs:
                    scores.append(usage_encoding.get_response(pair, user_prompt_template = 'Please tell me how similar the meaning of the word \'{target}\' is in the following example sentences: \n1. {usage_1}\n2. {usage_2}'))
            else:
                class WiCBinary(BaseModel):
                    change : bool = Field(description='Whether the word has the same meaning or not.')
                usage_encoding.structure = WiCBinary
                for pair in target_usage_pairs:
                    change_score = int(usage_encoding.get_response(pair, user_prompt_template = 'Please tell me if the meaning of the word \'{target}\' is the same in the following example sentences: \n1. {usage_1}\n2. {usage_2}'))
                    scores.append(change_score)
        
        return scores            


class GCDPipeline(Pipeline):
    def __init__(self):
        super().__init__()

    def get_scores(self, target_usages : List[Set[TargetUsage]],
                   usage_encoding,
                   metric : GradedChange):
        if isinstance(usage_encoding, PromptModel):
            pass # Is this possible?
        
        elif isinstance(usage_encoding, ChatModelDefinitionGenerator):
            encoded_usages_t1 = usage_encoding.generate_definitions(target_usages[0], encode=True)
            encoded_usages_t2 = usage_encoding.generate_definitions(target_usages[1], encode=True)

        elif isinstance(usage_encoding, ContextualizedModel):
            encoded_usages_t1 = usage_encoding.encode(target_usages[0])
            encoded_usages_t2 = usage_encoding.encode(target_usages[1])

        # Measure the change using the metric
        change = metric.compute_scores(encoded_usages_t1, encoded_usages_t2)
        return change
