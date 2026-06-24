"""Corpus utilities for line-level corpora and search helpers."""

import bz2
import gzip
import logging
import os
import re
from typing import List, Pattern, Self, Union
from itertools import islice
from collections import defaultdict, deque
from pathlib import Path
import lxml.etree as ET
from sortedcontainers import SortedKeyList
from inspect import signature
import json
import urllib

import spacy
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset
from datasets.dataset_dict import DatasetDict

from languagechange.resource_manager import LanguageChange
from languagechange.search import SearchTerm
from languagechange.usages import TargetUsage, TargetUsageList, UsageDictionary
from languagechange.utils import LiteralTime, PARSE_DATE_SIMPLE, PARSE_DATE_ADV
from languagechange.config import SPACY_PACKAGES_PATH

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SPACY_FEATURE_NAMES = {"token": "text", "lemma": "lemma_", "pos_tag": "pos_"}


class Line:
    """Wraps a corpus line with token, lemma, and POS metadata."""

    def __init__(self,
                 raw_text=None,
                 tokens=None,
                 lemmas=None,
                 pos_tags=None,
                 fname=None,
                 raw_lemma_text=None,
                 raw_pos_text=None,
                 **kwargs,
                 ):
        self._raw_text = raw_text
        self._raw_lemma_text = raw_lemma_text
        self._raw_pos_text = raw_pos_text
        self._tokens = tokens
        self._lemmas = lemmas
        self._pos_tags = pos_tags
        self._fname = fname
        self.__dict__.update(kwargs)

    def tokens(self):
        if not self._tokens == None:
            return self._tokens
        else:
            return self._lemmas

    def lemmas(self):
        return self._lemmas

    def pos_tags(self):
        return self._pos_tags

    def tokens_by_feature(self, feat=str):
        if feat == 'token':
            return self.tokens()
        elif feat == 'lemma':
            return self.lemmas()
        elif feat == 'pos_tag':
            return self.pos_tags()
        else:
            raise ValueError(f"'{feat}' is not a valid word feature")

    def raw_text(self):
        if not self._raw_text == None:
            return self._raw_text
        else:
            if not self._tokens == None:
                return ' '.join(self._tokens)
            elif not self._lemmas == None:
                return ' '.join(self._lemmas)
            else:
                raise Exception('No valid data in Line')

    def raw_lemma_text(self):
        if not self._raw_lemma_text == None:
            return self._raw_lemma_text
        return ' '.join(self._lemmas)

    def raw_pos_text(self):
        if not self._raw_pos_text == None:
            return self._raw_pos_text
        return ' '.join(self._pos_tags)

    def raw_text_by_feature(self, feat='token'):
        if feat == 'token':
            return self.raw_text()
        elif feat == 'lemma':
            return self.raw_lemma_text()
        elif feat == 'pos_tag':
            return self.raw_pos_text()
        else:
            raise ValueError(f"'{feat}' is not a valid word feature")

    def search(self, search_term: SearchTerm, time=None) -> TargetUsageList:
        """
            Searches the line given a search_term.

            Args:
                search_term : SearchTerm
            Returns: A TargetUsageList of all matches.
        """
        time = getattr(self, 'date', time)
        tul = TargetUsageList()

        if search_term.regex:

            def search_func(regexp, line):
                """
                    Finds all occurences in 'line' (possibly multiple lines) matching the 'regexp' (possibly multiple
                    words).
                """
                offsets = []
                rex = re.compile(fr'(?<!\S){regexp}(?!\S)', re.MULTILINE)
                for fi in re.finditer(rex, line):
                    offsets.append((fi.start(), fi.end()))
                return offsets

            def get_corresponding_token_offsets(feature, feature_offsets):
                """
                    Takes a 'feature' and target offsets for the raw feature string, and returns the corresponding
                    offsets for the raw token string.
                """
                token_offsets = []
                sorted_offsets = deque(sorted(feature_offsets, reverse=True))
                target_offsets = sorted_offsets.pop()
                feat_acc_chars, token_acc_chars = 0, 0
                start, stop = None, None
                for token, feat in zip(self.tokens() + [''], self.tokens_by_feature(feature) + ['']):
                    # Reached the end of the current target
                    if feat_acc_chars == target_offsets[1] + 1:
                        stop = token_acc_chars
                        if start is not None:
                            # Add the offsets and get new target offsets, if there are any
                            token_offsets.append((start, stop - 1))
                            start = None
                            stop = None
                            if not sorted_offsets:
                                break
                            target_offsets = sorted_offsets.pop()
                    # Reached the beginning of the current target
                    if feat_acc_chars == target_offsets[0]:
                        start = token_acc_chars
                    # Increase the accumulated character counts of tokens and the feature
                    feat_acc_chars += len(feat) + 1
                    token_acc_chars += len(token) + 1
                return token_offsets

            token_offsets = set()

            no_match = False
            for feat, term in search_term.feature_value_pairs.items():
                raw_text_by_feature = self.raw_text_by_feature(feat)
                feature_offsets = search_func(term, raw_text_by_feature)
                if feature_offsets:
                    corr_offsets = get_corresponding_token_offsets(feat, feature_offsets)
                    if len(token_offsets) == 0:
                        token_offsets = set(corr_offsets)
                    else:
                        # Only keep the offsets for which all features so far have returned a match
                        token_offsets = token_offsets.intersection(set(corr_offsets))
                else:
                    no_match = True
                    break

            if not no_match:
                for offsets in sorted(token_offsets):
                    tu = TargetUsage(self.raw_text(), list(offsets), time, id=getattr(self, 'id', 0))
                    tul.append(tu)
        else:
            for idx, values in enumerate(
                    zip(*(self.tokens_by_feature(f) for f in search_term.feature_value_pairs.keys()))):
                if list(values) == list(search_term.feature_value_pairs.values()):
                    offsets = [0, 0]
                    if not idx == 0:
                        offsets[0] = len(' '.join(self.tokens()[:idx])) + 1
                    offsets[1] = offsets[0] + len(self.tokens()[idx])
                    tu = TargetUsage(self.raw_text(), offsets, time, id=getattr(self, 'id', 0))
                    tul.append(tu)

        return tul

    def __str__(self):
        return self.raw_text()
    
    def __len__(self):
        return len(self.tokens())


class Corpus:
    """Base interface for corpora that support search and tokenization."""

    def __init__(self, name, language=None, time=LiteralTime('no time specification'),
                 time_function=None, skip_lines=0, **args):
        self.name = name
        self.language = language
        if time_function is not None and callable(time_function):
            self.time = time_function(self)
        elif hasattr(self, 'extract_dates') and callable(self.extract_dates):
            self.time = self.extract_dates()
        else:
            self.time = time
        self.skip_lines = skip_lines

    def search(self,
               search_terms: Union[str, Pattern, SearchTerm, List[Union[str, Pattern, SearchTerm]]],
               parse_date="simple"
               ) -> UsageDictionary:
        """
            Searches through the corpora by calling Line.search() on all lines.

            Args:
                search_terms : List[ str | Pattern | SearchTerm ]
                    If a search term is str or Pattern it is converted
                    to a SearchTerm and matches tokens only
                    SearchTerm(word_feature = 'token'). #TODO: change this
                parse_date (str, default='simple'): passed to `self.line_iterator`, if applicable.

            Returns: A UsageDictionary containing all search results for each search term.
        """
        if not isinstance(search_terms, list):
            search_terms = [search_terms]

        usage_dictionary = UsageDictionary()
        for st in search_terms:
            if not isinstance(st, SearchTerm):
                st = SearchTerm(st, regex=True if isinstance(st, Pattern) else False)
            tul = TargetUsageList()
            usage_dictionary[str(st)] = tul
            sig = signature(self.line_iterator)
            if "parse_date" in sig.parameters:
                it = self.line_iterator(parse_date=parse_date)
            else:
                it = self.line_iterator()
            for line in it:
                match: List[TargetUsage] = line.search(st, time=self.time)
                tul.extend(match)
        n_usages = sum(map(len, usage_dictionary.values()))
        logging.info(f"{n_usages} usages found.")
        return usage_dictionary

    def _initialize_nlp(self, package_name=None):
        """
        Initializes a spaCy NLP pipeline for self.language.
        """
        if package_name is None:
            with open(SPACY_PACKAGES_PATH) as f:
                spacy_packages = json.load(f)
                if self.language is not None:
                    if self.language.lower() not in spacy_packages:
                        logging.info(f"SpaCy does not have support for {self.language}. Falling back to multilingual processing.")
                        language = "xx"
                    else:
                        language = self.language.lower()
                else:
                    logging.info("No language is defined for the corpus. Falling back to multilingual processing.")
                    language = "xx"

                package_name = spacy_packages[language]
        nlp = spacy.load(package_name)
        return nlp

    def _process_text_iter(self, 
            nlp_model="spacy",
            package_name=None,
            features={"token", "lemma", "pos_tag"},
            split_sentences=False,
            sentencizer="spacy",
            sentencizer_batch_size=1024):
        """
        Processes the corpus by streaming through it and yields tokens, lemmas and POS tags in new Line objects.
        """
        if nlp_model == "spacy":
            nlp = self._initialize_nlp(package_name=package_name)
            required = {
                "lemma": "lemmatizer",
                "pos_tag": "tagger",
            }
            for feat in features:
                component = required.get(feat)
                if component and component not in nlp.pipe_names:
                    logging.error(f"SpaCy model '{nlp.meta.get('name')}' does not support feature '{feat}'.")
                    raise ValueError
        else:
            nlp = nlp_model

        if split_sentences:
            it = self._split_sentences_iter(sentencizer=sentencizer, package_name=package_name, batch_size=sentencizer_batch_size)
        else:
            it = self.line_iterator()

        raw_texts = filter(lambda s: isinstance(s, str) and s.strip(), (line.raw_text() for line in it))

        if nlp_model == "spacy":
            processed = nlp.pipe(raw_texts)
        else:
            processed = nlp(raw_texts)

        for text, line in zip(raw_texts, processed):
            token_level_features = {
                feat + "s": [
                    getattr(token, SPACY_FEATURE_NAMES[feat] if nlp_model == "spacy" else feat) for token in line] 
                for feat in features}
            line_features = {"raw_text": text} | token_level_features
            yield Line(**line_features)

    def _split_sentences_iter(self, sentencizer="spacy", package_name=None, batch_size=1024):
        """
        Iterates through the corpus and splits the text into one sentence per line.

        Args:
            sentencizer (Union[str, callable], default='spacy'): the sentencizer to use, by default the one of spaCy. 
                If not 'spacy', a function with the signature str -> [str], splitting a string into sentence strings in 
                a list, is expected.
            package_name (str, default=None): a spaCy package name, such as 'en_core_web_sm', that allows using a 
                specific spaCy NLP model. If not defined, the default model for the language of the corpus will be
                used.
            batch_size (int, default=1024): the amount of lines to process simultaneously when iterating through the 
                corpus and splitting into sentences.

        Yields:
            Line: a Line object containing a sentence.
        """
        if sentencizer == "spacy":
            sentencizer = self._initialize_nlp(package_name=package_name)
            sentencizer.add_pipe('sentencizer')
            lines = []
            for line in self.line_iterator():
                lines.append(line.raw_text())
                if len(lines) >= batch_size:
                    sentences = list(sentencizer(' '.join(lines)).sents)
                    # Do not yield the last sentence since it might be cut off
                    for sent in sentences[:-1]:
                        yield Line(sent.text)
                    lines = [sentences[-1].text]
            sentences = sentencizer(' '.join(lines)).sents
            for sent in sentences:
                yield Line(sent.text)

        elif callable(sentencizer):
            try:
                lines = []
                for line in self.line_iterator():
                    lines.append(line.raw_text())
                    if len(lines) >= batch_size:
                        sentences = list(sentencizer(' '.join(lines)))
                        # Do not yield the last sentence since it might be cut off
                        for sent in sentences[:-1]:
                            yield Line(sent)
                        lines = [sentences[-1]]
                sentences = sentencizer(' '.join(lines))
                for sent in sentences:
                    yield Line(sent)
            except:
                logging.info(f"ERROR: Could not use method {sentencizer} directly as a function to split sentences.")
    
    def process(
            self,
            save_format='parquet',
            file_specification=None,
            nlp_model="spacy",
            package_name=None,
            features=['token', 'lemma', 'pos_tag'],
            split_sentences=True,
            sentencizer="spacy",
            sentencizer_batch_size=1024
        ):
        """
        Tokenizes, lemmatizes and POS tags the corpus using a spaCy NLP pipeline for the language of the corpus. 
        Optionally, sentence splitting is also done. Saves the processed corpus to a new file and returns the corpus.

        Args:
            save_format (str, default='parquet'): the format to save the resulting corpus in. One of 'parquet', 
                'vertical' and 'linebyline'.
            file_specification (str, default=None): suffix to add to the files, by default adding each of the added 
                features to the file name. 
            nlp_model (Union[str, callable], default="spacy"): the NLP model to use for processing, by default spaCy.
                A custom model can also be used. In this case, a function taking a Iterable[str] (the sentences) and 
                returning an Iterable[List[object]], where each object in the inner list has attributes `token`, 
                `lemma` and `pos_tag`, is expected.
            package_name (str, default=None): a spaCy package name, such as 'en_core_web_sm', that allows using a 
                specific spaCy NLP model. If not defined, the default model for the language of the corpus will be
                used.
            features (list[str], default=['token', 'lemma', 'pos_tag']): the features to extract. Must be a subset of
                {'token', 'lemma', 'pos_tag'}.
            split_sentences (bool, default=False): whether to perform sentence splitting or not.
            sentencizer (Union[str, callable], default="spacy"): the sentencizer to use if sentence splitting (see 
                ``self._split_sentences_iter``).
            sentencizer_batch_size (int, default=1024): batch size for sentence splitting, passed to
                ``self._split_sentences_iter``.
        
        Returns:
            Corpus: the processed corpus.
        """
        if file_specification is None:
            file_specification = ""
            for feat in features:
                file_specification += f"-{feat}s"
        if save_format == "vertical" or save_format == "parquet":
            file_ending = ".tsv"
        elif save_format == "linebyline":
            file_ending = ".txt"
        else:
            logging.error("'save_format' must be one of 'vertical', 'parquet' and 'linebyline'")
            raise ValueError

        it = self._process_text_iter(
            nlp_model=nlp_model,
            package_name=package_name,
            features=features,
            split_sentences=split_sentences,
            sentencizer=sentencizer,
            sentencizer_batch_size=sentencizer_batch_size)

        f_path = os.path.splitext(self.path)[0] + file_specification + file_ending

        with open(f_path, 'w+') as f: # cache is probably needed here because the file might already exist.
            if save_format == 'linebyline':
                if not len(features) == 1:
                    logging.error("When saving processed data to a `LinebyLineCorpus`, only one feature can be stored")
                    raise ValueError
                feature = features[0]
                for line in it:
                    f.write(' '.join(line.tokens_by_feature(feature))+'\n')
                return LinebyLineCorpus(f_path, feature=feature)

            elif save_format == 'vertical' or save_format == 'parquet':
                field_separator = getattr(self, 'field_separator', '\t')
                sentence_separator = getattr(self, 'sentence_separator', '\n')

                def write_vertical_line(fields):
                    fields = [f for f in fields if f is not None]
                    for tup in zip(*fields):
                        f.write(field_separator.join(tup) + sentence_separator)
                    f.write(sentence_separator)
                
                for line in it:
                    write_vertical_line([line.tokens_by_feature(feat) for feat in features])
                
                tsv_corpus = VerticalCorpus(f_path, field_map={feat: i for i, feat in enumerate(features)})

                if save_format == 'parquet':
                    processed_corpus = tsv_corpus.cast_to_parquet()
                    os.remove(tsv_corpus.path)
                else:
                    processed_corpus = tsv_corpus
                return processed_corpus
    
    def tokenize(
            self,
            save_format='parquet',
            file_specification=None,
            nlp_model="spacy",
            package_name=None,
            split_sentences=True,
            sentencizer="spacy",
            sentencizer_batch_size=1024
        ):
        """
        Tokenizes the corpus using a spaCy NLP pipeline for the language of the corpus. Optionally, sentence splitting 
        is also done. Saves the processed corpus to a new file and returns the corpus.

        Args:
            save_format (str, default='parquet'): the format to save the resulting corpus in. One of 'parquet', 
                'vertical' and 'linebyline'.
            file_specification (str, default=None): suffix to add to the files, by default adding each of the added 
                features to the file name. 
            nlp_model (Union[str, callable], default="spacy"): the NLP model to use for processing, by default spaCy.
                A custom model can also be used. In this case, a function taking a Iterable[str] (the sentences) and 
                returning an Iterable[List[object]], where each object in the inner list has a 'token' attribute is 
                expected.
            package_name (str, default=None): a spaCy package name, such as 'en_core_web_sm', that allows using a 
                specific spaCy NLP model. If not defined, the default model for the language of the corpus will be
                used.
            split_sentences (bool, default=False): whether to perform sentence splitting or not.
            sentencizer (Union[str, callable], default="spacy"): the sentencizer to use if sentence splitting (see 
                ``self._split_sentences_iter``).
            sentencizer_batch_size (int, default=1024): batch size for sentence splitting, passed to
                ``self._split_sentences_iter``.
        
        Returns:
            Corpus: the processed corpus.
        """
        return self.process(
            save_format=save_format,
            file_specification=file_specification,
            nlp_model=nlp_model,
            package_name=package_name,
            features=['token'],
            split_sentences=split_sentences,
            sentencizer=sentencizer,
            sentencizer_batch_size=sentencizer_batch_size
        )

    def lemmatize(
            self,
            save_format='parquet',
            file_specification=None,
            nlp_model="spacy",
            package_name=None,
            split_sentences=True,
            sentencizer="spacy",
            sentencizer_batch_size=1024
        ):
        """
        Lemmatizes the corpus using a spaCy NLP pipeline for the language of the corpus. Optionally, sentence splitting 
        is also done. Saves the processed corpus to a new file and returns the corpus.

        Args:
            save_format (str, default='parquet'): the format to save the resulting corpus in. One of 'parquet', 
                'vertical' and 'linebyline'.
            file_specification (str, default=None): suffix to add to the files, by default adding each of the added 
                features to the file name. 
            nlp_model (Union[str, callable], default="spacy"): the NLP model to use for processing, by default spaCy.
                A custom model can also be used. In this case, a function taking a Iterable[str] (the sentences) and 
                returning an Iterable[List[object]], where each object in the inner list a `lemma` attribute is 
                expected.
            package_name (str, default=None): a spaCy package name, such as 'en_core_web_sm', that allows using a 
                specific spaCy NLP model. If not defined, the default model for the language of the corpus will be
                used.
            split_sentences (bool, default=False): whether to perform sentence splitting or not.
            sentencizer (Union[str, callable], default="spacy"): the sentencizer to use if sentence splitting (see 
                ``self._split_sentences_iter``).
            sentencizer_batch_size (int, default=1024): batch size for sentence splitting, passed to
                ``self._split_sentences_iter``.
        
        Returns:
            Corpus: the processed corpus.
        """
        return self.process(
            save_format=save_format,
            file_specification=file_specification,
            nlp_model=nlp_model,
            package_name=package_name,
            features=['lemma'],
            split_sentences=split_sentences,
            sentencizer=sentencizer,
            sentencizer_batch_size=sentencizer_batch_size
        )
    
    def pos_tag(
            self,
            save_format='parquet',
            file_specification=None,
            nlp_model="spacy",
            package_name=None,
            split_sentences=True,
            sentencizer="spacy",
            sentencizer_batch_size=1024
        ):
        """
        POS tags the corpus using a spaCy NLP pipeline for the language of the corpus. Optionally, sentence splitting 
        is also done. Saves the processed corpus to a new file and returns the corpus.

        Args:
            save_format (str, default='parquet'): the format to save the resulting corpus in. One of 'parquet', 
                'vertical' and 'linebyline'.
            file_specification (str, default=None): suffix to add to the files, by default adding each of the added 
                features to the file name. 
            nlp_model (Union[str, callable], default="spacy"): the NLP model to use for processing, by default spaCy.
                A custom model can also be used. In this case, a function taking a Iterable[str] (the sentences) and 
                returning an Iterable[List[object]], where each object in the inner list has a 'pos_tag' attribute is 
                expected.
            package_name (str, default=None): a spaCy package name, such as 'en_core_web_sm', that allows using a 
                specific spaCy NLP model. If not defined, the default model for the language of the corpus will be
                used.
            split_sentences (bool, default=False): whether to perform sentence splitting or not.
            sentencizer (Union[str, callable], default="spacy"): the sentencizer to use if sentence splitting (see 
                ``self._split_sentences_iter``).
            sentencizer_batch_size (int, default=1024): batch size for sentence splitting, passed to
                ``self._split_sentences_iter``.
        
        Returns:
            Corpus: the processed corpus.
        """
        return self.process(
            save_format=save_format,
            file_specification=file_specification,
            nlp_model=nlp_model,
            package_name=package_name,
            features=['pos_tag'],
            split_sentences=split_sentences,
            sentencizer=sentencizer,
            sentencizer_batch_size=sentencizer_batch_size
        )

    def folder_iterator(self, path): #TODO: integrate with the rest of the class, or remove

        fnames = []

        for fname in os.listdir(path):

            if os.path.isdir(os.path.join(path, fname)):
                fnames = fnames + self.folder_iterator(os.path.join(path, fname))
            else:
                fnames.append(os.path.join(path, fname))

        return fnames

    def cast_to_vertical(corpora, vertical_corpus): #TODO: remove or change

        line_iterators = [corpus.line_iterator() for corpus in corpora]
        iterate = True

        with open(vertical_corpus.path, 'w+') as f:

            while iterate:
                lines = []
                for iterator in line_iterators:
                    next_line = next(iterator)
                if not next_line == None:
                    vertical_lines = []
                    for j in range(len(lines[0])):
                        vertical_lines.append('{vertical_corpus.field_separator}'.join(
                            [lines[i][j] for i in range(len(lines))]))
                    for line in vertical_lines:
                        f.write(line+'\n')
                    f.write(vertical_corpus.sentence_separator)
                else:
                    iterate = False

    def save(self):
        lc = LanguageChange()
        lc.save_resource('corpus', f'{self.language} corpora', self.name)

    def __iter__(self):
        yield from self.line_iterator()


class LinebyLineCorpus(Corpus):

    def __init__(self, 
            path, 
            feature=None,
            is_sentence_split=False,
            tokens_splitter=None,
            **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = path
        super().__init__(**kwargs)
        self.path = path

        self.is_sentence_split = is_sentence_split or feature
        self.feature = feature
        self.tokens_splitter = tokens_splitter or ' '

    def line_iterator(self):

        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(line):
            line = line.replace('\n', '')
            data = {}
            data['raw_text'] = line
            data[f"{self.feature}s"] = line.split(self.tokens_splitter)
            return data

        for fname in fnames:

            if fname.endswith('.txt'):
                with open(fname, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= self.skip_lines:
                            data = get_data(line)
                            yield Line(fname=fname, **data)

            elif fname.endswith('.gz'):
                with gzip.open(fname, mode="rt") as f:
                    for i, line in enumerate(f):
                        if i >= self.skip_lines:
                            data = get_data(line)
                            yield Line(fname=fname, **data)

            else:
                raise Exception('Format not recognized')


class ParquetCorpus(Corpus):
    def __init__(self,
                 path: str,
                 load_from_huggingface_hub: bool = False,
                 token_level_features: set = {"token", "lemma", 'pos_tag'},
                 sentence_level_features: set = {"id", "date"},
                 column_names: dict = None,
                 **kwargs):
        """
        Initialize a ParquetCorpus.

        Args:
            path (str): path to a local Parquet file, a directory containing Parquet files, or a Hugging Face dataset 
                identifier.
            load_from_huggingface_hub (bool): if true, load the dataset from the Hugging Face Hub using `path`; 
                otherwise treat `path` as a local filesystem path.
            **args: additional keyword arguments passed to `Corpus`.
        """
        super().__init__(name=path, **kwargs)

        if load_from_huggingface_hub:
            # Download from huggingface hub
            self.dataset = load_dataset(path)
            self.path = None
        else:
            self.path = path
            self.dataset = None

        self.column_names = {f: f for f in token_level_features.union(sentence_level_features)}
        if column_names:
            self.column_names.update(column_names)

        self.rev_column_names = {name: feature for feature, name in self.column_names.items()}
        if not len(self.rev_column_names) == len(self.column_names):
            logging.error("Two features have the same name.")
            raise ValueError

        self.token_name = self.column_names["token"]
        self.id_name = self.column_names["id"]
        self.date_name = self.column_names["date"]

        self.token_level_features = token_level_features
        self.sentence_level_features = sentence_level_features

        self.token_level_feature_names = {self.column_names.get(f, f) for f in self.token_level_features}
        self.sentence_level_feature_names = {self.column_names.get(f, f) for f in self.sentence_level_features}

    def _get_iters(self, split=None):
        if self.path:
            if os.path.isdir(self.path):
                iters = self.folder_iterator(self.path)
            else:
                iters = [self.path]
        else:
            if isinstance(self.dataset, DatasetDict):
                if split is None:
                    iters = [self.dataset[k] for k in self.dataset.keys()]
                else:
                    iters = [self.dataset[split]]
            elif isinstance(self.dataset, Dataset):
                iters = [self.dataset]
            else:
                raise TypeError(f"{self.dataset} is neither a Dataset nor a DatasetDict.")
        return iters

    def line_iterator(self, chunk_size: int = 10 ** 6, split=None):
        """
        Iterate over the corpus and yield `Line` objects grouped by `id`.

        Token-level columns are converted to plural field names in the output (e.g. `token` -> `tokens`). Features in
        `self.sentence_level_features` are assumed to be constant within each line.

        Args:
            chunk_size (int): number of rows to load per batch.
            split (str|None): dataset split to use when reading from a Hugging
                Face dataset. If none, all splits are iterated.

        Yields:
            Line: a Line object containing token-level values as lists and
                constant metadata fields as scalars.
        """
        for it in self._get_iters(split=split):
            if self.path:
                parquet_file = pq.ParquetFile(it)
                iterator = parquet_file.iter_batches(batch_size=chunk_size)
            else:
                iterable = it.to_iterable_dataset()
                iterator = iterable.iter(batch_size=chunk_size)

            for batch in iterator:
                if self.path:
                    chunk = batch.to_pandas()
                else:
                    chunk = pd.DataFrame(batch)
                cols = chunk.columns
                constant_line_fields = set(cols).intersection(self.sentence_level_feature_names)
                variable_line_fields = list(set(cols).intersection(self.token_level_feature_names))
                for _, sent in chunk.groupby(self.id_name, sort=False):
                    yield Line(**{
                        f'{self.rev_column_names[k]}s': sent[k].tolist() for k in variable_line_fields
                    } | {
                        self.rev_column_names[k]: sent[k].iloc[0] for k in constant_line_fields
                    })

    def search(self,
               search_terms: Union[str, Pattern, SearchTerm, List[Union[str, Pattern, SearchTerm]]],
               parse_date="simple",
               chunk_size: int = 10 ** 6,
               split=None
               ):
        """
        Search the corpus for token-level matches against any of the search terms and return the target usages 
        consisting of the lines they occur in, with the tokens(s) highlighted.

        Each search term is matched against feature (token/lemma/pos) rows according to the values in the search term. 
        For each match, the full sentence or line is reconstructed from all rows sharing the same `id`, and character 
        offsets are computed for the matched token.

        Args:
            search_terms (List[str|Pattern|SearchTerm]): search terms to apply. Strings and regex patterns are 
                converted to `SearchTerm` instances automatically.
            parse_date (str, default='simple'): how to parse dates (and times). If 'none', the dates are only converted 
                to strings. If 'simple', the dates are assumed to be in ISO format, and only YYYY-MM-DD are kept by
                cutting the string. If 'advanced', uses pd.to_datetime to parse dates (flexible but slow).
            chunk_size (int): number of rows to load per batch.
            split (str|None): dataset split to use when reading from a Hugging Face dataset. If none, all splits are 
                searched.

        Returns:
            UsageDictionary: a dictionary of `TargetUsageList` objects for each search term.
        """
        if not isinstance(search_terms, list):
            search_terms = [search_terms]
        if parse_date is None or parse_date == "none":
            def parse_date(d):
                return d.apply(str)
        elif parse_date == "simple":
            def parse_date(d):
                return d.apply(str).apply(PARSE_DATE_SIMPLE)
        elif parse_date == "advanced":
            def parse_date(d):
                return (pd.to_datetime(d.apply(str)).dt.tz_localize(None).dt.strftime("%Y-%m-%d"))
        else:
            raise ValueError("`parse_date` must be one of ['none', 'simple' and 'advanced'].")

        usages = defaultdict(list)

        for i, st in enumerate(search_terms):
            if not isinstance(st, SearchTerm):
                st = SearchTerm(st, regex=True if isinstance(st, Pattern) else False)
                search_terms[i] = st

        for it in self._get_iters(split=split):
            if self.path:
                parquet_file = pq.ParquetFile(it)
                iterator = parquet_file.iter_batches(batch_size=chunk_size)
            else:
                iterable = it.to_iterable_dataset()
                iterator = iterable.iter(batch_size=chunk_size)

            start_index = 0
            for batch in iterator:
                if self.path:
                    chunk = batch.to_pandas()
                else:
                    chunk = pd.DataFrame(batch)

                chunk.index = range(start_index, start_index + len(chunk))
                start_index += len(chunk)

                chunk_occurrences = chunk.copy()
                all_matches = False
                chunk_occurrences["target"] = ""
                for st in search_terms:
                    term_matches = True
                    features_values = st.feature_value_pairs.items()
                    for f, v in features_values:
                        if st.regex:
                            term_matches &= chunk_occurrences[self.column_names[f]].str.fullmatch(v, na=False)
                        else:
                            term_matches &= chunk_occurrences[self.column_names[f]] == v
                    all_matches |= term_matches
                    chunk_occurrences.loc[term_matches, "target"] = str(st)

                chunk_occurrences = chunk_occurrences[all_matches]

                if chunk_occurrences.empty:
                    continue

                ids = chunk_occurrences[self.id_name].unique()
                chunk_tokens = chunk[chunk[self.id_name].isin(ids)].copy()
                if chunk_tokens.empty:
                    continue

                chunk_tokens.sort_index(inplace=True)

                chunk_tokens['length'] = chunk_tokens[self.token_name].str.len()
                chunk_tokens['space'] = 1
                is_last = chunk_tokens[self.id_name] != chunk_tokens[self.id_name].shift(-1)
                chunk_tokens.loc[is_last, 'space'] = 0
                chunk_tokens['start'] = (
                    chunk_tokens.groupby(self.id_name)[['length', 'space']].cumsum()['length'] -
                    chunk_tokens['length'] +
                    chunk_tokens.groupby(self.id_name)['space'].cumsum() -
                    chunk_tokens['space']
                )
                chunk_tokens['end'] = chunk_tokens['start'] + chunk_tokens['length']
                chunk_tokens = chunk_tokens.drop(columns=['length', 'space'])
                chunk_tokens = chunk_tokens.astype({'start': 'Int32', 'end': 'Int32'})

                chunk_sentences = (
                    chunk_tokens
                    .groupby(self.id_name)[self.token_name]
                    .apply(lambda x: x.str.cat(sep=" "))
                    .reset_index()
                    .rename(columns={self.token_name: 'text'})
                )

                chunk_targets = chunk_occurrences.merge(
                    chunk_tokens[['start', 'end']], left_index=True, right_index=True
                )
                chunk_targets = chunk_targets.merge(chunk_sentences, on=self.id_name)
                if self.date_name in chunk_targets:
                    chunk_targets[self.date_name] = parse_date(chunk_targets[self.date_name])
                chunk_targets['offsets'] = (chunk_targets[['start', 'end']]
                                            .apply(lambda row: row.values, axis=1)
                                            .apply(lambda row: row.tolist()))

                chunk_targets = (
                    chunk_targets.
                    drop(columns=['start', 'end', self.id_name]).
                    rename(columns={self.date_name: "time"})
                )

                if "time" in chunk_targets:
                    chunk_targets["time"] = chunk_targets["time"].apply(LiteralTime)

                # Add to usage dictionary
                for target, group in chunk_targets.groupby("target"):
                    tul = [TargetUsage(**tu) for tu in group.to_dict("records")]
                    usages[target].extend(tul)

        usage_dictionary = UsageDictionary({t: TargetUsageList(tul) for t, tul in usages.items()})
        n_usages = sum(map(len, usage_dictionary.values()))
        logging.info(f"{n_usages} usages found.")

        return usage_dictionary

    def push_to_hub(self, repo_name=None, private=False):
        """
        Push the corpus to the Hugging Face Hub.

        Args:
            repo_name (str|None): name of the destination repository. If None and the corpus has a local path, the stem
                of that path is used.
            private (bool): if true, push the dataset to a private repository.
        """
        if repo_name is None and self.path:
            repo_name = Path(self.path).stem
        if self.path:
            local_dataset = Dataset.from_parquet(self.path)
        else:
            local_dataset = self.dataset
        local_dataset.push_to_hub(repo_id=repo_name, private=private)


class VerticalCorpus(Corpus):

    def __init__(self, path, sentence_separator='\n', field_separator='\t',
                 field_map={'token': 0, 'lemma': 1, 'pos_tag': 2},
                 has_header=False, **args):
        super().__init__(name=path, **args)
        self.path = path
        self.sentence_separator = sentence_separator
        self.field_separator = field_separator
        self.field_map = field_map
        self.has_header = has_header

    def line_iterator(self):

        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(splitted_line):
            data = {}
            # Token level features
            for field in {'token', 'lemma', 'pos_tag'}.intersection(self.field_map.keys()):
                text = [vertical_line[field] for vertical_line in splitted_line]
                data[f'{field}s'] = text
            if 'tokens' in data:
                data['raw_text'] = ' '.join(data['tokens'])
            # Sentence level features
            for field in {'id', 'date'}.intersection(self.field_map.keys()):
                for i in range(1, len(splitted_line)):
                    assert splitted_line[i][field] == splitted_line[0][field], f"found two values \
                        '{splitted_line[i][field]}' and '{splitted_line[0][field]}' for the same feature \
                        {field} in a line."
                text = splitted_line[0][field]
                data[field] = text
            return data

        def iterate_lines(f):
            line = []
            # If the corpus has a header describing the columns, this overrides self.field_map
            if self.has_header:
                field_map = {k: i for i, k in enumerate(next(iter(f)).strip("\n").split(self.field_separator))}
            else:
                field_map = self.field_map
            for vertical_line in islice(f, self.skip_lines, None):
                if vertical_line.strip(self.field_separator) == self.sentence_separator:
                    data = get_data(line)
                    yield Line(fname=fname, **data)
                    line = []
                else:
                    fields = vertical_line.strip('\n').split(self.field_separator)
                    line.append({field: fields[field_map[field]] for field in field_map})

        for fname in fnames:
            if fname.endswith('.txt') or fname.endswith('.csv') or fname.endswith('.tsv'):
                with open(fname, 'r') as f:
                    yield from iterate_lines(f)

            elif fname.endswith('.gz'):
                with gzip.open(fname, mode="rt") as f:
                    yield from iterate_lines(f)

            else:
                raise Exception('Format not recognized')

    def cast_to_parquet(self,
                        parquet_corpus_or_path: Union[ParquetCorpus, str] = None,
                        block_size: int = 1 << 20,
                        compression: str = "snappy",
                        delimiter: str = "\t",
                        add_sentence_ids: bool = True,
                        **kwargs) -> None:
        if parquet_corpus_or_path is None:
            extensions = "".join(Path(self.path).suffixes)
            pq_name = str(self.path).replace(extensions, ".parquet")
            parquet_corpus = ParquetCorpus(Path(pq_name), **kwargs)
        elif isinstance(parquet_corpus_or_path, str):
            parquet_corpus = ParquetCorpus(Path(parquet_corpus_or_path), **kwargs)
        elif isinstance(parquet_corpus_or_path, ParquetCorpus):
            parquet_corpus = parquet_corpus_or_path
        else:
            raise TypeError("'parquet_corpus_or_path' has to be one of [None, str, ParquetCorpus].")
        parquet_path = parquet_corpus.path

        read_opts = pv.ReadOptions(block_size=block_size, autogenerate_column_names=not self.has_header)
        parse_opts = pv.ParseOptions(delimiter=delimiter, quote_char=False)
        convert_opts = pv.ConvertOptions()

        writer = None

        if add_sentence_ids:
            # Iterate through all sentences and add ids to separate them
            cols = list(kv[0] for kv in sorted(self.field_map.items(), key=lambda kv: kv[1]))
            for i, line in enumerate(self.line_iterator()):
                table = pa.table({k: pa.array(line.tokens_by_feature(k)) for k in cols} | {"id": [i] * len(line)})
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema, compression=compression)
                writer.write_table(table)
                del line

        else:
            csv_reader = pv.open_csv(self.path, read_options=read_opts,
                                    convert_options=convert_opts, parse_options=parse_opts)

            for batch in csv_reader:
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, batch.schema, compression=compression)
                writer.write_table(table)

        if writer:
            writer.close()
        
        return parquet_corpus


# Should be able to load and parse a corpus in XML format.
# Supports only tokenized corpora so far.
class XMLCorpus(Corpus):

    def __init__(self, path, sentence_tag='sentence', token_tag='token', is_lemmatized=False, lemma_tag=None,
                 is_pos_tagged=False, pos_tag_tag=None, text_tag='text', **kwargs):
        if not 'name' in kwargs:
            name = path
        super().__init__(name, **kwargs)
        self.path = path

        if lemma_tag:
            self.lemma_tag = lemma_tag
        else:
            self.lemma_tag = ''

        if is_lemmatized:
            self.is_lemmatized = True
            if lemma_tag != '':
                self.lemma_tag = lemma_tag
            else:
                self.lemma_tag = 'lemma'
        else:
            self.is_lemmatized = False
            self.lemma_tag = ''

        if pos_tag_tag:
            self.pos_tag_tag = pos_tag_tag
        else:
            self.pos_tag_tag = ''

        if is_pos_tagged:
            self.is_pos_tagged = True
            if pos_tag_tag != '':
                self.pos_tag_tag = pos_tag_tag
            else:
                self.pos_tag_tag = 'pos_tag'
        else:
            self.is_pos_tagged = False
            self.pos_tag_tag = ''

        self.sentence_tag = sentence_tag
        self.token_tag = token_tag
        self.text_tag = text_tag

    def get_attribute(self, tag, attribute):
        return tag.attrib[attribute]

    def line_iterator(self, parse_date="simple"):
        """
        Iterates through all sentences in the XML file.

        Args:
            parse_date (str, default='simple'): how to parse dates (and times). If 'none', the dates are only converted 
                to strings. If 'simple', the dates are assumed to be in ISO format, and only YYYY-MM-DD are kept by
                cutting the string. If 'advanced', uses pd.to_datetime to parse dates (flexible but slow).
        """
        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        if parse_date is None or parse_date == "none":
            parse_date = str
        elif parse_date == "simple":
            parse_date = PARSE_DATE_SIMPLE
        elif parse_date == "advanced":
            parse_date = PARSE_DATE_ADV
        else:
            raise ValueError("`parse_date` must be one of ['none', 'simple' and 'advanced'].")

        def get_data(tokens, lemmas=[], pos_tags=[]):
            data = {}
            data['raw_text'] = ' '.join(tokens)
            if self.is_lemmatized and lemmas != []:
                data['lemmas'] = lemmas
            if self.is_pos_tagged and pos_tags != []:
                data['pos_tags'] = pos_tags
            data['tokens'] = tokens
            return data

        def read_xml(source):
            tokens = []
            lemmas = []
            pos_tags = []
            curr_id = None
            curr_date = None
            curr_time = None
            parser = ET.iterparse(source, events=('start', 'end'))

            for event, elem in parser:
                if elem.sourceline >= self.skip_lines:
                    if elem.tag == self.text_tag:
                        new_date = elem.get('date', None)
                        if new_date != curr_date:
                            curr_date = new_date
                            curr_time = LiteralTime(parse_date(curr_date)) if curr_date is not None else None
                    if elem.tag == self.sentence_tag:
                        if event == 'start':
                            sentence_id = elem.get('id', None)
                            # A new sentence
                            if sentence_id != curr_id:
                                if tokens:
                                    data = get_data(tokens, lemmas, pos_tags)
                                    if curr_time is not None:
                                        data['date'] = curr_time
                                    data['id'] = curr_id
                                    yield Line(fname=fname, **data)
                                curr_id = sentence_id
                                tokens = []
                                lemmas = []
                                pos_tags = []

                        elif event == 'end':
                            elem.clear()
                    elif elem.tag == self.token_tag:
                        if event == 'end':
                            if self.is_lemmatized:
                                lemma = self.get_attribute(elem, self.lemma_tag)
                                lemmas.append(lemma)
                            if self.is_pos_tagged:
                                pos_tag = self.get_attribute(elem, self.pos_tag_tag)
                                pos_tags.append(pos_tag)
                            token = elem.text.strip()
                            tokens.append(token)
                            elem.clear()
                    else:
                        if event == 'end':
                            elem.clear()
            # Yield the last sentence
            if tokens:
                data = get_data(tokens, lemmas, pos_tags)
                data['date'] = curr_time
                line_id = sentence_id
                data['id'] = line_id
                yield Line(fname=fname, **data)
                elem.clear()

            del parser

        for fname in fnames:
            if fname.endswith('.xml'):
                for l in read_xml(fname):
                    yield l
            elif fname.endswith('.xml.bz2'):
                with bz2.open(fname, 'r') as f:
                    for l in read_xml(f):
                        yield l
            else:
                raise Exception('Format not recognized')

    # Cast to a LineByLine corpus and save the result in the path specified in there
    def cast_to_linebyline(self, linebyline_corpus_or_path=None):
        if linebyline_corpus_or_path is None:
            basename = os.path.splitext(self.path)[0]
            linebyline_corpus = LinebyLineCorpus(basename + ".txt")
        elif isinstance(linebyline_corpus_or_path, str):
            linebyline_corpus = LinebyLineCorpus(linebyline_corpus_or_path)
        elif isinstance(linebyline_corpus_or_path, LinebyLineCorpus):
            linebyline_corpus = linebyline_corpus_or_path
        else:
            raise TypeError("'linebyline_corpus_or_path' has to be one of [None, str, LinebyLineCorpus].")
        savepath = linebyline_corpus.path
        if hasattr(linebyline_corpus, 'tokens_splitter'):
            tokens_splitter = linebyline_corpus.tokens_splitter
        else:
            tokens_splitter = ' '
        tokenized = linebyline_corpus.is_tokenized
        lemmatized = linebyline_corpus.is_lemmatized
        if lemmatized and not self.is_lemmatized:
            logging.info('ERROR: cannot cast to lemmatized LinebyLineCorpus because this XMLCorpus is not lemmatized.')
            return None
        with open(savepath, 'w+') as f:
            if lemmatized:
                for line in self.line_iterator():
                    f.write(tokens_splitter.join(line.lemmas())+'\n')  # cache needed here
            elif tokenized:
                for line in self.line_iterator():
                    f.write(tokens_splitter.join(line.tokens())+'\n')  # cache needed here
            else:
                for line in self.line_iterator():
                    f.write(line.raw_text()+'\n')  # cache needed here

        return linebyline_corpus

    def cast_to_vertical(self, vertical_corpus_or_path=None, write_header=True):
        if vertical_corpus_or_path is None:
            extensions = "".join(Path(self.path).suffixes)
            vrt_name = str(self.path).replace(extensions, ".tsv")
            vertical_corpus = VerticalCorpus(
                vrt_name,
                field_map={k: i for i, k in enumerate(["token", "lemma", 'pos_tag', "id", "date"])})
        elif isinstance(vertical_corpus_or_path, str):
            vertical_corpus = VerticalCorpus(
                vertical_corpus_or_path,
                field_map={k: i for i, k in enumerate(["token", "lemma", 'pos_tag', "id", "date"])})
        elif isinstance(vertical_corpus_or_path, VerticalCorpus):
            vertical_corpus = vertical_corpus_or_path
        else:
            raise TypeError("'vertical_corpus_or_path' has to be one of [None, str, VerticalCorpus].")
        savepath = vertical_corpus.path
        field_separator = vertical_corpus.field_separator
        sentence_separator = vertical_corpus.sentence_separator
        # We need to make sure that the line features (token, lemma, pos, etc.) come in the same order as in the
        # field_map in the vertical_corpus
        sorted_field_names = [key for (key, _) in sorted(vertical_corpus.field_map.items(), key=lambda x: x[1])]
        # id and date are the same across tokens, lemmas etc. in a sentence
        constant_line_fields = set(sorted_field_names).intersection({'id', 'date'})
        variable_line_fields = list(set(sorted_field_names).difference({'id', 'date'}))

        def get_line_feature(line, key):
            field_name_to_line_feature = {
                'token': line.tokens(),
                'lemma': line.lemmas(),
                'pos_tag': line.pos_tags(),
                'id': line.id,
                'date': str(line.date)}
            return field_name_to_line_feature[key]

        with open(savepath, 'w', newline='') as csvfile:
            if write_header:
                csvfile.write(field_separator.join(sorted_field_names) + "\n")
            for line in self.line_iterator():
                sentence_content = {k: get_line_feature(line, k) for k in constant_line_fields}
                for values in zip(*(get_line_feature(line, k) for k in variable_line_fields)):
                    token_content = {f: v for f, v in zip(variable_line_fields, values)}
                    all_content = sentence_content | token_content
                    vertical_row = [all_content[s] for s in sorted_field_names]
                    csvfile.write(field_separator.join(vertical_row) + "\n")
                if sentence_separator is not None:
                    csvfile.write(sentence_separator)
        
        return vertical_corpus

    def cast_to_parquet(self, parquet_corpus_or_path=None, keep_tsv=False):
        extensions = "".join(Path(self.path).suffixes)
        vrt_name = str(self.path).replace(extensions, ".tsv")
        c = VerticalCorpus(
            vrt_name,
            field_map={k: i for i, k in enumerate(["token", "lemma", 'pos_tag', "id", "date"])},
            sentence_separator=None,
            has_header=True)
        logging.info("Casting to intermediate tsv file...")
        self.cast_to_vertical(c)
        logging.info("Casting to parquet...")
        parquet_corpus = c.cast_to_parquet(
            parquet_corpus_or_path=parquet_corpus_or_path,
            add_sentence_ids=False)
        if not keep_tsv:
            os.remove(c.path)
            logging.info(f"Removed temporary file {c.path}.")
        return parquet_corpus


# A class for handling XML corpora specifically from spraakbanken.gu.se
class SprakBankenCorpus(XMLCorpus):

    def __init__(
            self, path, sentence_tag='sentence', token_tag='token', is_lemmatized=True, lemma_tag='lemma',
            is_pos_tagged=True, pos_tag_tag='pos', **args):
        super().__init__(path, sentence_tag, token_tag, is_lemmatized, lemma_tag, is_pos_tagged, pos_tag_tag, **args)

    def get_attribute(self, tag, attribute):
        content = tag.attrib[attribute]
        if content != None:
            if attribute == self.lemma_tag:
                content = content.strip("|").split("|")
                if content != ['']:
                    return content[0]
            else:
                return content
        return tag.text.strip()


class HistoricalCorpus(SortedKeyList):

    def __new__(cls, *args, **kwargs):
        """Ensures only valid arguments go to SortedKeyList"""
        return super().__new__(cls)

    def __init__(self, corpora: Union[List[Corpus], str], key=lambda c: c.time, corpus_type=None, time_function=None):
        """
            This class is a SortedKeyList of corpora. A historical corpus can be initialised either from a path where the files are located, or from a list of already instanciated Corpus objects.

            Args:
                corpora ([Corpus]|str): a list of corpora or a path where the corpora are stored.
                key (function, default = lambda c : c.time): the key by which the corpora are sorted. Default sorting is by time, in ascending order
                corpus_type (str, default=None): the kind of corpus. Needs to be provided if initalising from a folder, and then needs to be one of 'line_by_line','vertical','xml', and 'sprakbanken'.
                time_function (function, default = None): the function used to extract a time value for each corpus. Needed if initialising from a folder.
        """
        if isinstance(corpora, str):
            try:
                if corpus_type not in ['line_by_line', 'vertical', 'xml', 'sprakbanken']:
                    logging.error(
                        "When initialising from a folder path, corpus_type must be one of 'line_by_line','vertical','xml' and 'sprakbanken'.")
                    raise ValueError
                corpora_list = []
                for file in os.listdir(corpora):
                    try:
                        if corpus_type == 'line_by_line':
                            corpus = LinebyLineCorpus(os.path.join(corpora, file), time_function=time_function)
                        elif corpus_type == 'vertical':
                            corpus = VerticalCorpus(os.path.join(corpora, file), time_function=time_function)
                        elif corpus_type == 'xml':
                            corpus = XMLCorpus(os.path.join(corpora, file), time_function=time_function)
                        elif corpus_type == 'sprakbanken':
                            corpus = SprakBankenCorpus(os.path.join(corpora, file), time_function=time_function)
                        corpora_list.append(corpus)
                    except:  # TODO: proper exception
                        logging.error(f"Could not initialise a corpus from path {os.path.join(dir,file)}.")
                        continue
                corpora = corpora_list
            except:
                logging.error(f"Could not use path {corpora} to intitialize corpora.")
                raise Exception
        elif isinstance(corpora, list):
            for corpus in corpora:
                if not isinstance(corpus, Corpus):
                    logging.error("Every element in 'corpora' needs to be a Corpus object.")
                    raise Exception
        else:
            logging.error("'corpora' needs to be either a string or a list of Corpus objects.")
            raise Exception
        super().__init__(corpora, key)

    def line_iterator(self):
        """
            Iterates through all of the corpora, and yields all of the lines that are possible to get.
        """
        for corpus in self:
            try:
                for line in corpus.line_iterator():
                    yield line
            except:
                logging.error(f"Could not get lines from {corpus.name}.")

    def search(self, search_terms: List[str | Pattern | SearchTerm], index_by_corpus=False):
        """
            Searches through all of the corpora by calling search() for each of them.

            Args:
                search_terms : List[ str | Pattern | SearchTerm ]
                    If search term is str or Pattern it is converted
                    to a SearchTerm and matches tokens only
                    SearchTerm(word_feature = 'token').
                index_by_corpus : bool, default False
                    decides whether the usages for a given word should be a dictionary,
                    with keys as the corpus names and values as lists of usages, or a list
                    of all usages across corpora.

            Returns: a dictionary containing all search results from the included corpora.
        """

        if index_by_corpus:
            usages = {}
            for corpus in self:
                try:
                    usage_dict: UsageDictionary = corpus.search(search_terms)
                except:
                    logging.error(f"Could not search through {corpus.name}.")
                    continue
                for key in usage_dict:
                    if not key in usages:
                        usages[key] = {str(corpus.name): TargetUsageList()}
                    usages[key][str(corpus.name)] = usage_dict[key]

        else:
            usages = UsageDictionary()
            for corpus in self:
                try:
                    usage_dict: UsageDictionary = corpus.search(search_terms)
                except:
                    logging.error(f"Could not search through {str(corpus.name)}.")
                    continue
                for key in usage_dict:
                    if not key in usages:
                        usages[key] = TargetUsageList()
                    usages[key].extend(usage_dict[key])

        return usages
