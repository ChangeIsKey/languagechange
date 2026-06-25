"""Simple time representations used across the LanguageChange toolkit."""

from typing import Union
from numbers import Number
import math
import json
import logging

import numpy as np
import matplotlib as mpl
import pandas as pd
import spacy

from languagechange.config import SPACY_PACKAGES_PATH


def PARSE_DATE_SIMPLE(d): return d[:10]


def PARSE_DATE_ADV(d): return pd.to_datetime(d).strftime("%Y-%m-%d")



class Time:
    def __init__(self):
        pass


class LiteralTime(Time):
    """Represents a literal timestamp or label for usage references."""

    def __init__(self, time: str):
        self.time = time

    def __eq__(self, other):
        assert type(other) == LiteralTime
        return self.time == other.time

    def __lt__(self, other):
        assert type(other) == LiteralTime
        return self.time < other.time

    def __le__(self, other):
        assert type(other) == LiteralTime
        return self.time <= other.time

    def __repr__(self):
        return self.time


class NumericalTime(Time):
    """Numeric timestamp (e.g., time slice) that supports comparisons."""

    def __init__(self, time: Number):
        self.time = time

    def __eq__(self, other):
        if type(other) == NumericalTime:
            return self.time == other.time
        elif type(other) == TimeInterval:
            return self.time == other.start.time

    # todo: what if the other is a literal time?
    def __lt__(self, other):
        if type(other) == NumericalTime:
            return self.time < other.time
        elif type(other) == TimeInterval:
            return self.time < other.start.time

    def __le__(self, other):
        if type(other) == NumericalTime:
            return self.time <= other.time
        elif type(other) == TimeInterval:
            return self.time <= other.start.time

    def __repr__(self):
        return str(self.time)


class TimeInterval(Time):
    """Represents an interval between two Time points."""

    def __init__(self, start: Time, end: Time):
        self.start = start
        self.end = end
        if type(self.start).__name__ == type(self.end).__name__:
            if type(self.start) == NumericalTime:
                self.duration = self.end.time - self.start.time
        else:
            raise Exception('start and end points have to be of the same type')

    def __eq__(self, other):
        assert type(other) == TimeInterval
        return self.start == other.start and self.end == other.end

    # todo: what if the other is a literal time?
    def __lt__(self, other):
        if type(other) == TimeInterval:
            if self.start == other.start:
                return self.duration < other.duration
            else:
                return self.start < other.start
        elif type(other) == NumericalTime:
            return self.start.time < other.time

    def __le__(self, other):
        if type(other) == TimeInterval:
            if self.start == other.start:
                return self.duration <= other.duration
            else:
                return self.start <= other.start
        elif type(other) == NumericalTime:
            return self.start.time <= other.time

    def __repr__(self):
        return f"{self.start.time} - {self.end.time}"


def _parse_year(time : Union[str, int]):
    """
        Takes a string or Time describing a date and tries to parse it and return the year.
    """
    if isinstance(time, int):
        return time
    try:
        parsed = pd.to_datetime(str(time))
        return parsed.year
    except ValueError as e:
        logging.error(f"Could not parse the date '{str(time)}' due to {e}")
        raise e


def generate_colormap(n_classes):
    """
        Generate a colormap with 'n_classes' colors that are distinguishable from each other, along with grey, useful 
        for clustering.

        Args:
            n_classes (int): the amount of unique classes, excluding outliers (no class)

        Returns:
            cmap (matplotlib.colors.ListedColormap): a colormap mapping 0 to grey and c to a unique color, for all c in 
                range(1, n_classes+1).
    """
    num_hues = math.ceil(n_classes / 3)
    hues = (np.arange(num_hues) / num_hues)[np.arange(n_classes) // 3]
    saturations = np.full(n_classes, 1)
    values = np.tile(np.linspace(0.6, 1, 3), n_classes // 3 + 1)[:n_classes]
    hsv = np.stack([hues, saturations, values], axis=1)
    including_grey = np.vstack(([(0.7, 0.7, 0.7)], mpl.colors.hsv_to_rgb(hsv)))
    cmap = mpl.colors.ListedColormap(including_grey)
    return cmap


def _initialize_nlp(language, package_name=None):
        """
        Initializes a spaCy NLP pipeline for language.
        """
        if package_name is None:
            with open(SPACY_PACKAGES_PATH) as f:
                spacy_packages = json.load(f)
                if language is not None:
                    if language.lower() not in spacy_packages:
                        logging.info(f"SpaCy does not have support for {language}. Falling back to multilingual processing.")
                        language = "xx"
                    else:
                        language = language.lower()
                else:
                    logging.info("No language is defined for the corpus. Falling back to multilingual processing.")
                    language = "xx"

                package_name = spacy_packages[language]
        nlp = spacy.load(package_name)
        return nlp
