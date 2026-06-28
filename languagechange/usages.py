"""Target usage helpers and containers for LanguageChange."""

import enum
import pickle
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Union, Tuple

import jsonlines
import numpy as np

from languagechange.utils import Time, LiteralTime, NumericalTime, TimeInterval, _parse_year


class POS(enum.Enum):
    """Enumeration of supported parts of speech for targets."""

    NOUN = 1
    VERB = 2
    ADJECTIVE = 3
    ADVERB = 4


class Target:
    """Stores a target word together with optional metadata."""

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
    """Represents an individual usage with offsets and optional time metadata."""

    def __init__(self, text: str, offsets: str, time: Union[Time, str, int] = None, **kwargs):
        self.text_ = text
        self.offsets = offsets
        if isinstance(time, str):
            self.time = LiteralTime(time)
        elif isinstance(time, int):
            self.time = NumericalTime(time)
        elif isinstance(time, Time):
            self.time = time
        elif time is not None:
            logging.error("'time' has to be a NumericalTime, LiteralTime, str, int or None.")
            raise TypeError
        else:
            time = self.time
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
        d = self.__dict__.copy()
        d['time'] = str(d['time'])
        return d

    def __getitem__(self, item):
        return self.text_[item]

    def __str__(self):
        return self.text_


class DWUGUsage(TargetUsage):
    """DWUG-specific usage metadata, including annotator judgments."""

    def __init__(self, target, date, grouping, identifier, description,  **args):
        super().__init__(**args)
        self.target = target
        self.date = date
        self.grouping = grouping
        self.identifier = identifier
        self.description = description

    def to_dict(self):
        d = self.__dict__.copy()
        d['time'] = str(d['time'])
        d['target'] = str(d['target'])
        return d

class TargetUsageList(list):
    """List of TargetUsage instances with serialization helpers."""

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

    def group_by_interval(self,
            intervals : list[Union[TimeInterval, Tuple[str, str], Tuple[int, int]]],
            time_attr="time",
            use_year=False):
        """
        Group usages by time interval.

        Each usage is assigned to the interval whose boundaries contain the value of ``time_attr`` (by default, 
        'time'). The result is returned as a UsageDictionary whose keys are string representations of the intervals.

        Args:
            intervals (list[Union[TimeInterval, Tuple[str, str], Tuple[int, int]]]): time intervals used for grouping.
            time_attr (str, default="time"): name of the usage attribute containing temporal information.
            use_year (bool, default=False): if True, groups usages by year. Otherwise, exact dates are used.

        Returns:
            UsageDictionary: mapping from interval strings to TargetUsageList objects containing the usages that fall 
                within each interval.
        """
        if not all(getattr(u, time_attr, None) for u in self):
            logging.error(f"In order to sort by '{time_attr}', all usages need to have a '{time_attr}' attribute.")
            raise AttributeError
        if not isinstance(intervals, list):
            logging.error("`intervals` has to be a list of TimeInterval or tuples of str or int.")
            raise TypeError
        for i, interval in enumerate(intervals):
            if not isinstance(interval, TimeInterval):                    
                interval = TimeInterval(*(t if isinstance(t, Time) else LiteralTime(str(t)) for t in interval))
            if use_year:
                interval = TimeInterval(*(LiteralTime(str(_parse_year(t))) for t in (interval.start, interval.end)))
            intervals[i] = interval

        def get_time(u):
            u_t = getattr(u, time_attr)
            if not isinstance(u_t, Time) or use_year:
                u_t = LiteralTime(str(_parse_year(u_t)))
            return u_t

        ins = deque(sorted(intervals))
        us = deque(sorted(self, key = get_time))
        usage_dict = UsageDictionary()
        t = ins.popleft()
        u = us.popleft()
        u_t = get_time(u)
        curr_u = TargetUsageList()

        while u and t:
            if u_t <= t.end:
                if u_t >= t.start:
                    curr_u.append(u)

                if us:
                    u = us.popleft()
                    u_t = get_time(u)
                else:
                    u = None
            else:
                if curr_u:
                    t_rep = str(t.start) if t.start == t.end else str(t)
                    usage_dict[t_rep] = curr_u

                if ins:
                    t = ins.popleft()
                    curr_u = TargetUsageList()
                else:
                    t = None
        if curr_u and t:
            t_rep = str(t.start) if t.start == t.end else str(t)
            usage_dict[t_rep] = curr_u
        
        return usage_dict

    def group_by_time(self, times : list[Union[Time, str, int]] = None, time_attr="time", use_year=True):
        """
        Group usages by time point.

        If ``times`` is not provided, usages are grouped by year inferred from ``time_attr`` (by default, 'time'). 
        Otherwise, usages are grouped according to the supplied time points, by exact matching.

        Args:
            times (list[Union[Time, str, int]], optional): explicit time points to group by.
            time_attr (str, default="time"): name of the usage attribute containing temporal information.
            use_year (bool, default=True): if True, groups usages by year. Otherwise, exact time values are used.

        Returns:
            UsageDictionary: mapping from time labels to TargetUsageList objects.
        """
        if times and not isinstance(times, list):
            logging.error("`times` has to be a list of Time, str or int.")
            raise TypeError
        if times is None:
            if use_year:
                sorted_times = sorted(set(_parse_year(getattr(u, time_attr)) for u in self))
            else:
                sorted_times = sorted(set(str(getattr(u, time_attr)) for u in self))
            intervals = [TimeInterval(LiteralTime(str(y)), LiteralTime(str(y))) for y in sorted_times]
        else:
            if all(isinstance(t, str) or isinstance(t, int) for t in times):
                times = [LiteralTime(str(t)) for t in times]
            intervals = [TimeInterval(t, t) for t in times]
        return self.group_by_interval(intervals, time_attr=time_attr, use_year=use_year)

    def _sample(self, generator, n_samples):
        if n_samples < len(self) and n_samples != 0:
            return generator.choice(self, size=n_samples, replace=False).tolist()
        return self

    def _group_and_sample(self, groups, grouping_fn, n_samples=0, random_seed=None, time_attr="time", use_year=False):
        rng = np.random.default_rng(seed=random_seed)
        sampled = TargetUsageList()
        usages_by_group = grouping_fn(groups, time_attr=time_attr, use_year=use_year)
        for _, g_usages in sorted(usages_by_group.items(), key = lambda i : i[0]):
            sampled.extend(g_usages._sample(rng, n_samples))
        return sampled

    def sample_per_interval(self, intervals, n_samples=0, random_seed=None, time_attr="time", use_year=False):
        """
        Sample usages independently from each time interval.

        Usages are first grouped by interval and then up to ``n_samples`` usages are sampled without replacement from 
        each group.

        Args:
            intervals (list[TimeInterval]): time intervals used for grouping.
            n_samples (int, default=0): maximum number of usages to sample per interval. If ``0`` or greater than the 
                group size, all usages are retained.
            random_seed (int, optional): seed used for reproducible sampling.
            time_attr (str, default="time"): name of the usage attribute containing temporal information.
            use_year (bool, default=True): if True, groups usages by year. Otherwise, exact time values are used.

        Returns:
            TargetUsageList: sampled usages from all intervals combined into a single list.
        """
        return self._group_and_sample(
            intervals, 
            self.group_by_interval,
            n_samples=n_samples,
            random_seed=random_seed,
            time_attr=time_attr,
            use_year=use_year)

    def sample_per_time(self, times=None, n_samples=0, random_seed=None, time_attr="time", use_year=True):
        """
        Sample usages independently from each time point.

        Usages are first grouped by time and then up to ``n_samples`` usages are sampled without replacement from each 
        group.

        Args:
            times (list[Time], optional): explicit time points to group by. If omitted, usages are grouped by inferred 
                years.
            n_samples (int, default=0): maximum number of usages to sample per time point. If ``0`` or greater than the 
                group size, all usages are retained.
            random_seed (int, optional): seed used for reproducible sampling.
            time_attr (str, default="time"): name of the usage attribute containing temporal information.
            use_year (bool, default=True): if True, groups usages by year. Otherwise, exact time values are used.

        Returns:
            TargetUsageList: sampled usages from all time points combined into a single list.
        """
        return self._group_and_sample(
            times, 
            self.group_by_time,
            n_samples=n_samples,
            random_seed=random_seed,
            time_attr=time_attr,
            use_year=use_year)
                

class UsageDictionary(dict):
    """Dictionary mapping words to TargetUsageList instances."""
    def __init__(self, *args, **kwargs):
        super().__init__()

        data = dict(*args, **kwargs)
        for k, v in data.items():
            super().__setitem__(k, v if isinstance(v, TargetUsageList) else TargetUsageList(v))

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
                        self[key] = TargetUsageList()
                        for tu in reader:
                            time = tu["time"]
                            if time is not None:
                                if isinstance(time, str):
                                    tu["time"] = LiteralTime(time)
                                elif isinstance(time, int):
                                    tu["time"] = NumericalTime(time)
                            self[key].append(TargetUsage(**tu))
                        logging.info(f"Loaded usages from {os.path.join(path, fn)}")
        not_loaded_words = words.difference(set(self.keys()))
        if len(not_loaded_words) != 0:
            logging.info(f"Could not find usages for words {not_loaded_words}")
