import enum
import pickle
import logging
import os
import re

import jsonlines

from pathlib import Path

from languagechange.utils import Time


class POS(enum.Enum):
   NOUN = 1
   VERB = 2
   ADJECTIVE = 3
   ADVERB = 4


class Target:
    def __init__(self, target : str):
        self.target = target

    def set_lemma(self, lemma: str):
        self.lemma = lemma

    def set_pos(self, pos:POS):
        self.pos = pos

    def __str__(self):
        return self.target

    def __hash__(self):
        return hash(self.target)


class TargetUsage:
    def __init__(self, text: str, offsets: str, time: Time = None, **kwargs):
        self.text_ = text
        self.offsets = offsets
        self.time = time
        self.__dict__.update(kwargs)

    def text(self):
        return self.text_

    def start(self):
        return self.offsets[0]

    def end(self):
        return self.offsets[1]

    def time(self):
        return self.time

    def to_dict(self):
        d = self.__dict__
        d['time'] = str(d['time'])
        return d

    def __getitem__(self,item):
        return self.text_[item]

    def __str__(self):
        return self.text_


class DWUGUsage(TargetUsage):

    def __init__(self, target, date, grouping, identifier, description,  **args):
        super().__init__(**args)
        self.target = target
        self.date = date
        self.grouping = grouping
        self.identifier = identifier
        self.description = description


class TargetUsageList(list):

    def save(self, path, target):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path,target), 'wb+') as f:
            pickle.dump(self,f)

    def load(path, target):
        with open(os.path.join(path,target),'rb') as f:
            return pickle.load(f)

    def time_axis(self):
        return [usage.time for usage in self]

    def to_dict(self):
        return [tu.to_dict() for tu in self]


class UsageDictionary(dict):

    def save(self, path, words = {}):
        Path(path).mkdir(parents=True, exist_ok=True)

        if words == {}:
            words = set(self.keys())
        else:
            words = set(words)
        words_not_present = words.difference(set(self.keys()))
        if len(words_not_present) != 0:
            logging.info(f'Words {words_not_present} are not in the usage dictionary')
        
        for k in set(self.keys()).intersection(words):
            output_fn = f"{path}/{k}_usages.jsonl"
            with jsonlines.open(output_fn, 'w') as writer:
                tul = self[k].to_dict()
                for i, tu in enumerate(tul):
                    tul[i] = {'text': tu['text_']} | tu # replace the 'text_' key with a 'text' key
                    tul[i].pop('text_')
                writer.write_all(tul)
                logging.info(f"Usages written to {output_fn}")

    def load(self, path, words = set()):
        if not os.path.exists(path):
            logging.error(f'Path {path} does not exist.')
            return None
        self.clear()
        words = set(words)
        for fn in os.listdir(path):
            match = re.findall(r'(.*)_usages\.jsonl', fn)
            if len(match) != 0:
                key = match[0]
                if key in words or len(words) == 0:
                    with jsonlines.open(os.path.join(path, fn), 'r') as reader:
                        self[key] = TargetUsageList(TargetUsage(**tu) for tu in reader)
                        logging.info(f"Loaded usages from {os.path.join(path, fn)}")
        not_loaded_words = words.difference(set(self.keys()))
        if len(not_loaded_words) != 0:
            logging.info(f"Could not find usages for words {not_loaded_words}")