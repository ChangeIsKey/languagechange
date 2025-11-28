import bz2
import gzip
import logging
import os
import re
from typing import List, Pattern, Self, Union

import lxml.etree as ET
import trankit
from languagechange.resource_manager import LanguageChange
from languagechange.search import SearchTerm
from languagechange.usages import TargetUsage, TargetUsageList, UsageDictionary
from languagechange.utils import LiteralTime
from sortedcontainers import SortedKeyList

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Line:

    def __init__(self,
                 raw_text=None,
                 tokens=None,
                 lemmas=None,
                 pos_tags=None,
                 fname=None,
                 raw_lemma_text=None,
                 raw_pos_text = None,
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

    def tokens_by_feature(self, feat = str):
        if feat == 'token':
            return self.tokens()
        elif feat == 'lemma':
            return self.lemmas()
        elif feat == 'pos':
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
        if not self._raw_lemmas == None:
            return self._raw_lemmas
        return ' '.join(self._lemmas)

    def raw_pos_text(self):
        if not self._raw_pos_text == None:
            return self._raw_pos_text
        return ' '.join(self._raw_pos_text)

    def raw_text_by_feature(self, feat = 'token'):
        if feat == 'token':
            return self.raw_text()
        elif feat == 'lemma':
            return self.raw_lemma_text()
        elif feat == 'pos':
            return self.raw_pos_text()
        else:
            raise ValueError(f"'{feat}' is not a valid word feature")

    def search(self, search_term : SearchTerm, time = None) -> TargetUsageList:
        """
            Searches the line given a search_term.

            Args:
                search_term : SearchTerm
            Returns: A TargetUsageList of all matches.
        """
        time  = getattr(self, 'date', time)
        tul = TargetUsageList()
        for feat in search_term.word_feature:
            if search_term.regex:
                if search_term.search_func:
                    def search_func(word, line):
                        offsets = []
                        rex = re.compile(f'( |^)+{word}( |$)+',re.MULTILINE)
                        for fi in re.finditer(rex, line):
                            s = line[fi.start():fi.end()].find(word)
                            offsets.append([fi.start()+s, fi.start()+s+len(word)])
                        return offsets
                raw_text_by_feature = self.raw_text_by_feature(feat)
                for offsets in search_func(search_term.term, raw_text_by_feature):
                    tu = TargetUsage(self.raw_text(), offsets, time, id=getattr(self, 'id', 0))
                    tul.append(tu)
            else:
                token_features = self.tokens_by_feature(feat)
                for idx, token in enumerate(token_features):
                    if search_term.term == token:
                        offsets = [0,0]
                        if not idx == 0:
                            offsets[0] = len(' '.join(self.tokens()[:idx])) + 1
                        offsets[1] = offsets[0] + len(self.tokens()[idx])
                        tu = TargetUsage(self.raw_text(), offsets, time, id=getattr(self, 'id', 0))
                        tul.append(tu)
        return tul

    def __str__(self):
        return self._raw_text


class Corpus:

    def __init__(self, name, language=None, time=LiteralTime('no time specification'), time_function = None, skip_lines=0, **args):
        self.name = name
        self.language = language
        if time_function is not None and callable(time_function):
            self.time = time_function(self)
        elif hasattr(self,'extract_dates') and callable(self.extract_dates):
            self.time = self.extract_dates()
        else:
            self.time = time
        self.skip_lines = skip_lines

    def set_sentences_iterator(self, sentences):
        self.sentences_iterator = sentences

    def search(self,
               search_terms: List[ str | Pattern | SearchTerm ]
               ) -> UsageDictionary:
        """
            Searches through the corpora by calling Line.search() on all lines.

            Args:
                search_terms : List[ str | Pattern | SearchTerm ]
                    If a search term is str or Pattern it is converted
                    to a SearchTerm and matches tokens only
                    SearchTerm(word_feature = 'token').

            Returns: A UsageDictionary containing all search results for each search term.
        """

        usage_dictionary = UsageDictionary()
        n_usages = 0
        for st in search_terms:
            if not isinstance(st, SearchTerm):
                st = SearchTerm(st, regex = True if isinstance(st, Pattern) else False)
            tul = TargetUsageList()
            usage_dictionary[st.term] = tul
            for line in self.line_iterator():
                match : List[TargetUsage] = line.search(st, time = self.time)
                tul.extend(match)
                n_usages += len(match)
        logging.info(f"{n_usages} usages found.")
        return usage_dictionary

    def tokenize(self, tokenizer = "trankit", split_sentences=False, batch_size=128):
        if tokenizer == "trankit":
            p = trankit.Pipeline(self.language)

            if split_sentences:

                def process_lines(texts):
                    tokenized = p.tokenize(' '.join(texts))
                    for sentence in tokenized['sentences']:
                        yield Line(raw_text=sentence['text'], tokens=[token['text'] for token in sentence['tokens']])

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_lines(texts):
                            yield line
                        texts = []
                if texts != []:
                    for line in process_lines(texts):
                        yield line  
                        
            else:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        tokenized_sentence = p.tokenize(text, is_sent=True)
                        line._tokens = [token['text'] for token in tokenized_sentence['tokens']]
                        yield line
        
        else:
            if hasattr(tokenizer, "tokenize") and callable(getattr(tokenizer,"tokenize")):
                tokenizer = tokenizer.tokenize
            
            if callable(tokenizer):
                try:
                    for line in self.line_iterator():
                        text = line.raw_text()
                        if type(text) == str and len(text.strip()) > 0:
                            line._tokens = [str(token) for token in tokenizer(text)]
                            yield line
                except Exception:
                    logging.error(f"Could not use tokenizer {tokenizer} directly as a function to tokenize.")

    def lemmatize(self, lemmatizer = "trankit", pretokenized = False, tokenize = False, split_sentences = False, batch_size=128):
        if lemmatizer == "trankit":
            p = trankit.Pipeline(self.language)

            # input which is not sentence split
            if split_sentences:
                
                def process_texts(texts):
                    lemmatized = p.lemmatize(' '.join(texts))
                    lines = []
                    for sentence in lemmatized['sentences']:
                        lines.append(Line(raw_text=sentence['text'], lemmas=[token['lemma'] for token in sentence['tokens']], tokens=[token['text'] for token in sentence['tokens']] if tokenize else None))
                    return lines

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_texts(texts):
                            yield line
                        texts = []

                if texts != []:
                    for line in process_texts(texts):
                        yield line

            # input which is not pretokenized, but each line is its own sentence
            elif not pretokenized:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        lemmatized_sentence = p.lemmatize(text, is_sent = True)
                        line._lemmas = [token['lemma'] for token in lemmatized_sentence['tokens']]
                        yield line

            # pretokenized input, one or more sentences at a time
            else:

                def modify_lines(lines):
                    lemmatized = p.lemmatize([line.tokens() for line in lines])
                    lemmatized_sentences = lemmatized['sentences']
                    for i, line in enumerate(lines):
                        line._lemmas = [token['lemma'] for token in lemmatized_sentences[i]['tokens']]
                        yield line

                lines = []
                for line in self.line_iterator():
                    tokens = line.tokens()
                    if type(tokens) == list and len(tokens) > 0:
                        lines.append(line)
                    if len(lines) == batch_size:
                        for line in modify_lines(lines):
                            yield line
                        lines = []
                if lines != []:
                    for line in modify_lines(lines):
                        yield line
                        

        # todo: add other lemmatizers if needed
        else:

            if hasattr(lemmatizer, "lemmatize") and callable(getattr(lemmatizer,"lemmatize")):
                lemmatizer = lemmatizer.lemmatize

            if callable(lemmatizer):
                try:
                    if pretokenized:
                        for line in self.line_iterator():
                            tokens = line.tokens()
                            if type(tokens) == list and len(tokens) != 0:
                                line._lemmas = [str(lemma) for lemma in lemmatizer(tokens)]
                                yield line
                    else:
                        for line in self.line_iterator():
                            text = line.raw_text()
                            if type(text) == str and len(text.strip()) > 0:
                                line._lemmas = [str(lemma) for lemma in lemmatizer(text)]
                                yield line
                except Exception:
                    logging.error(f"Could not use method {lemmatizer} directly as a function to lemmatize.")

    def pos_tagging(self, pos_tagger = "trankit", pretokenized = False, tokenize=False, split_sentences = False, batch_size=128):
        if pos_tagger == "trankit":
            p = trankit.Pipeline(self.language)

            # input which is not sentence split
            if split_sentences:

                def process_texts(texts):
                    pos_tagged = p.posdep(' '.join(texts))
                    for sentence in pos_tagged['sentences']:
                        yield Line(raw_text=sentence['text'], pos_tags=[token['upos'] for token in sentence['tokens']], tokens=[token['text'] for token in sentence['tokens']] if tokenize else None)

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_texts(texts):
                            yield line
                        texts = []

                if texts != []:
                    for line in process_texts(texts):
                        yield line

            # input which is not pretokenized, but each line is its own sentence
            elif not pretokenized:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        pos_tagged_sentence = p.posdep(text, is_sent = True)
                        line._pos_tags = [token['upos'] for token in pos_tagged_sentence['tokens']]
                        if tokenize:
                            line._tokens = [token['text'] for token in pos_tagged_sentence['tokens']]
                        yield line

            # pretokenized input, one or more sentences at a time
            else:

                def modify_lines(lines):
                    pos_tagged = p.posdep([line.tokens() for line in lines])
                    pos_tagged_sentences = pos_tagged['sentences']
                    for i, line in enumerate(lines):
                        line._pos_tags = [token['upos'] for token in pos_tagged_sentences[i]['tokens']]
                        yield line

                lines = []
                for line in self.line_iterator():
                    tokens = line.tokens()
                    if type(tokens) == list and len(tokens) > 0:
                        lines.append(line)
                    if len(lines) == batch_size:
                        for line in modify_lines(lines):
                            yield line
                        lines = []
                        
                if lines != []:
                    for line in modify_lines(lines):
                        yield line
                    
        else:
            if hasattr(pos_tagger, "pos_tag") and callable(getattr(pos_tagger,"pos_tag")):
                pos_tagger = pos_tagger.pos_tag
            if callable(pos_tagger):
                try:
                    if pretokenized:
                        for line in self.line_iterator():
                            tokens = line.tokens()
                            if type(tokens) == list and len(tokens) > 0:
                                line._pos_tags = [str(pos_tag) for pos_tag in pos_tagger(tokens)]
                                yield line

                    else:
                        for line in self.line_iterator():
                            text = line.raw_text()
                            if type(text) == str and len(text.strip()) > 0:
                                line._pos_tags = [str(pos_tag) for pos_tag in pos_tagger(text)]
                                yield line
                except Exception:
                    logging.error(f"Could not use method {pos_tagger} directly as a function to perform POS tagging.")

    def tokens_lemmas_pos_tags(self, nlp_model="trankit", tokens=True, split_sentences = False, batch_size=128):
        if nlp_model == "trankit":
            p = trankit.Pipeline(self.language)

            if not split_sentences:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        lemmatized_sentence = p.lemmatize(text, is_sent = True)
                        line._lemmas = [token['lemma'] for token in lemmatized_sentence['tokens']]
                        if tokens:
                            line._tokens = [token['text'] for token in lemmatized_sentence['tokens']]
                            pos_tagged = p.posdep(line.tokens(), is_sent=True)
                        else:
                            pos_tagged = p.posdep(line.raw_text(), is_sent=True)
                        line._pos_tags = [token['upos'] for token in pos_tagged['tokens']]
                        yield line

            else:

                def process_texts(texts):
                    lemmatized_sentences = p.lemmatize(' '.join(texts))
                    tokens = []
                    for sentence in lemmatized_sentences['sentences']:
                        tokens.append([token['text'] for token in sentence['tokens']])
                    pos_tagged_sentences = p.posdep(tokens)
                    for i, sentence in enumerate(lemmatized_sentences['sentences']):
                        yield Line(raw_text=sentence['text'], tokens=[token['text'] for token in sentence['tokens']] if tokens else None, lemmas=[token['lemma'] for token in sentence['tokens']],pos_tags=[token['upos'] for token in pos_tagged_sentences['sentences'][i]['tokens']])

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_texts(texts):
                            yield line
                        texts = []
                if len(texts) != 0:
                    for line in process_texts(texts):
                        yield line

    # preliminary function
    def segment_sentences(self, segmentizer = "trankit", batch_size=128):
        if segmentizer == "trankit":
            p = trankit.Pipeline(self.language)

            lines = []
            for line in self.line_iterator():
                lines.append(line.raw_text())
                if len(lines) == batch_size:
                    sentences = p.ssplit(' '.join(lines))
                    for sent in sentences['sentences']:
                        yield Line(sent['text'])
                    lines = []
            if len(lines) != 0:
                sentences = p.ssplit(' '.join(lines))
                for sent in sentences['sentences']:
                    yield Line(sent['text'])

        elif callable(segmentizer):
            try:
                lines = []
                for line in self.line_iterator():
                    lines.append(line.raw_text())
                    if len(lines) == batch_size:
                        sentences = segmentizer(' '.join(lines))
                        for sent in sentences:
                            yield Line(sent)
                        lines = []
                if len(lines) != 0:
                    sentences = segmentizer(' '.join(lines))
                    for sent in sentences:
                        yield Line(sent)
            except:
                logging.info(f"ERROR: Could not use method {segmentizer} directly as a function to split sentences.")


    def folder_iterator(self, path):

        fnames = []

        for fname in os.listdir(path):

            if os.path.isdir(os.path.join(path,fname)):
                fnames = fnames + self.folder_iterator(os.path.join(path,fname))
            else:
                fnames.append(os.path.join(path,fname))

        return fnames


    def cast_to_vertical(corpora, vertical_corpus):

        line_iterators = [corpus.line_iterator() for corpus in corpora]
        iterate = True

        with open(vertical_corpus.path,'w+') as f:

            while iterate:
                lines = []
                for iterator in line_iterators:
                    next_line = next(iterator)
                if not next_line == None:
                    vertical_lines = []
                    for j in range(len(lines[0])):
                        vertical_lines.append('{vertical_corpus.field_separator}'.join([lines[i][j] for i in range(len(lines))]))
                    for line in vertical_lines:
                        f.write(line+'\n')
                    f.write(vertical_corpus.sentence_separator)
                else:
                    iterate = False

    def save(self):
        lc = LanguageChange()
        lc.save_resource('corpus',f'{self.language} corpora',self.name)

    def save_tokenized_corpora(corpora : Union[Self, List[Self]], tokens = True, lemmas = False, pos = False, save_format = 'linebyline', file_specification = None, file_ending = ".txt", tokenizer="trankit", lemmatizer="trankit", pos_tagger="trankit", split_sentences = True, batch_size=128):
        if not type(corpora) is list:
            corpora = [corpora]
        if file_specification == None:
            file_specification = ""
            file_specification += "-tokens" if tokens else '' 
            file_specification += '-lemmas' if lemmas else '' 
            file_specification += '-pos' if pos else ''
        for corpus in corpora:
            tokenized_name = os.path.splitext(corpus.path)[0]+file_specification+file_ending
            with open(tokenized_name, 'w+') as f: # cache is probably needed here because the file might already exist.
                if save_format == 'linebyline':
                    if tokens:
                        for line in corpus.tokenize(tokenizer, split_sentences=split_sentences, batch_size=batch_size):
                            f.write(' '.join(line.tokens())+'\n') 
                    elif lemmas:
                        for line in corpus.lemmatize(lemmatizer, split_sentences=split_sentences, batch_size=batch_size):
                            f.write(' '.join(line.lemmas())+'\n')
                    elif pos:
                        for line in corpus.pos_tagging(pos_tagger,split_sentences=split_sentences, batch_size=batch_size):
                            f.write(' '.join(line.pos_tags())+'\n')
                elif save_format == 'vertical':

                    def write_vertical_line(fields):
                        fields = [f for f in fields if f is not None]
                        for tup in zip(*fields):
                            f.write('\t'.join(tup) + '\n')
                        f.write('\n')

                    if lemmas:
                        if pos:
                            # tokens_lemmas_pos (with or without tokens)
                            for line in corpus.tokens_lemmas_pos_tags(tokenizer, tokens=tokens,split_sentences=split_sentences, batch_size=batch_size):
                                write_vertical_line([line.tokens(), line.lemmas(), line.pos_tags()])
    
                        else:
                            # lemmatize (with or without tokens)
                            for line in corpus.lemmatize(lemmatizer, tokenize=tokens,split_sentences=split_sentences, batch_size=batch_size):
                                write_vertical_line([line.tokens(), line.lemmas(), line.pos_tags()])

                    elif pos:
                        # pos_tagging (with or without tokens)
                        for line in corpus.pos_tagging(pos_tagger, tokenize=tokens, split_sentences=split_sentences, batch_size=batch_size):
                            write_vertical_line([line.tokens(), line.lemmas(), line.pos_tags()])

                    elif tokens:
                        # tokenize only
                        for line in corpus.tokenize(tokenizer,split_sentences=split_sentences, batch_size=batch_size):
                            write_vertical_line([line.tokens(), line.lemmas(), line.pos_tags()])


class LinebyLineCorpus(Corpus):

    def __init__(self, path, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = path
        super().__init__(**kwargs)
        self.path = path

        if 'is_sentence_tokenized' in kwargs:
            self.is_sentence_tokenized = kwargs['is_sentence_tokenized']
        else:
            self.is_sentence_tokenized = False

        if self.is_sentence_tokenized:
            if 'is_tokenized' in kwargs:
                self.is_tokenized = kwargs['is_tokenized']
        else:
            if 'is_tokenized' in kwargs and kwargs['is_tokenized']:
                self.is_sentence_tokenized = True
                self.is_tokenized = True
            else:
                self.is_sentence_tokenized = False
                self.is_tokenized = False

        if 'is_tokenized' in kwargs and kwargs['is_tokenized']:
            if 'is_lemmatized' in kwargs:
                self.is_lemmatized = kwargs['is_lemmatized']
            if 'tokens_splitter' in kwargs:
                self.tokens_splitter = kwargs.tokens_splitter
            else:
                self.tokens_splitter = ' '
        else:
            if 'is_lemmatized' in kwargs and kwargs['is_lemmatized']:
                self.is_sentence_tokenized = True
                self.is_tokenized = True
                self.is_lemmatized = True
                if 'tokens_splitter' in kwargs:
                    self.tokens_splitter = kwargs.tokens_splitter
                else:
                    self.tokens_splitter = ' '
            else:
                self.is_lemmatized = False

    def line_iterator(self):
        
        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(line):
            line = line.replace('\n','')
            data = {}
            data['raw_text'] = line
            if self.is_lemmatized:
                data['lemmas'] = line.split(self.tokens_splitter)
            elif self.is_tokenized:
                data['tokens'] = line.split(self.tokens_splitter)
            return data

        for fname in fnames:

            if fname.endswith('.txt'):
                with open(fname,'r') as f:
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


class VerticalCorpus(Corpus):

    def __init__(self, path, sentence_separator='\n', field_separator='\t', field_map={'token':0, 'lemma':1, 'pos_tag':2}, **args):
        super().__init__(name=path,**args)
        self.path = path
        self.sentence_separator = sentence_separator
        self.field_separator = field_separator
        self.field_map = field_map

    def line_iterator(self):
        
        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(line):
            data = {}
            splitted_line = [vertical_line.strip('\n').split(self.field_separator) for vertical_line in line]
            raw_text = [vertical_line[self.field_map['token']] for vertical_line in splitted_line]
            data['raw_text'] = ' '.join(raw_text)
            data['tokens'] = raw_text
            if 'lemma' in self.field_map:
                lemma_text = [vertical_line[self.field_map['lemma']] for vertical_line in splitted_line]
                data['lemmas'] = lemma_text
            if 'pos_tag' in self.field_map:
                pos_text = [vertical_line[self.field_map['pos_tag']] for vertical_line in splitted_line]   
                data['pos_tags'] = pos_text
            return data

        for fname in fnames:

            if fname.endswith('.txt'):
                with open(fname,'r') as f:
                    line = []
                    for i, vertical_line in enumerate(f):
                        if i >= self.skip_lines:
                            if vertical_line == self.sentence_separator:
                                data = get_data(line)
                                yield Line(fname=fname, **data)
                                line = []
                            else:
                                line.append(vertical_line)

            elif fname.endswith('.gz'):
                with gzip.open(fname, mode="rt") as f:
                    for i, vertical_line in enumerate(f):
                        if i >= self.skip_lines:
                            if vertical_line == self.sentence_separator:
                                data = get_data(line)
                                yield Line(fname=fname, **data)
                                line = []
                            else:
                                line.append(vertical_line)

            else:
                raise Exception('Format not recognized')
            

# Should be able to load and parse a corpus in XML format.
# Supports only tokenized corpora so far.
class XMLCorpus(Corpus):

    def __init__(self, path, sentence_tag='sentence', token_tag='token', is_lemmatized=False, lemma_tag=None, is_pos_tagged=False, pos_tag_tag=None, text_tag='text', **args):
        if not 'name' in args:
            name = path
        super().__init__(name, **args)
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
                self.pos_tag_tag = 'pos'
        else:
            self.is_pos_tagged = False
            self.pos_tag_tag = ''

        self.sentence_tag = sentence_tag
        self.token_tag = token_tag
        self.text_tag = text_tag

    
    def get_attribute(self, tag, attribute):
        return tag.attrib[attribute]


    def line_iterator(self):
        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(tokens, lemmas = [], pos_tags = []):
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
            parser = ET.iterparse(source, events=('start','end'))
            sentence_counter = 0
            for event, elem in parser:
                if elem.sourceline >= self.skip_lines:
                    if elem.tag == self.text_tag:
                        date = elem.get('date')
                    if elem.tag == self.sentence_tag:
                        if event == 'start':
                            tokens = []
                            lemmas = []
                            pos_tags = []
                        # If the sentence has ended, create a new Line object with its content
                        elif event == 'end':
                            if tokens != []:
                                data = get_data(tokens, lemmas, pos_tags)
                                data['date'] = date
                                line_id = elem.get('id', sentence_counter)
                                data['id'] = line_id
                                yield Line(fname=fname, **data)
                                elem.clear()
                        sentence_counter += 1
                    elif elem.tag == self.token_tag:
                        if event == 'end':
                            if self.is_lemmatized:
                                lemma = self.get_attribute(elem, self.lemma_tag)
                                lemmas.append(lemma)
                            if self.is_pos_tagged:
                                pos_tag = self.get_attribute(elem, self.pos_tag_tag)
                                pos_tags.append(pos_tag)
                            token = elem.text
                            tokens.append(token)
                            elem.clear()
                    else:
                        if event == 'end':
                            elem.clear()


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
    def cast_to_linebyline(self, linebyline_corpus : LinebyLineCorpus):
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

    def cast_to_vertical(self, vertical_corpus : VerticalCorpus):
        savepath = vertical_corpus.path
        field_separator = vertical_corpus.field_separator
        sentence_separator = vertical_corpus.sentence_separator
        # We need to make sure that the line features (token, lemma, pos, etc.) come in the same order as in the field_map in the vertical_corpus
        sorted_field_names = [key for (key, _) in sorted(vertical_corpus.field_map.items(), key = lambda x : x[1])]
        
        def get_line_feature(line, key):
            field_name_to_line_feature = {'token': line.tokens, 'lemma': line.lemmas, 'pos_tag': line.pos_tags}
            return field_name_to_line_feature[key]()
        
        with open(savepath,'w+') as f:
            for line in self.line_iterator():
                for t in zip(*(get_line_feature(line, key) for key in sorted_field_names)):
                    f.write(field_separator.join(list(t))+'\n') # cache needed here
                f.write(sentence_separator) # cache needed here


# A class for handling XML corpora specifically from spraakbanken.gu.se
class SprakBankenCorpus(XMLCorpus):

    def __init__(self, path, sentence_tag='sentence',token_tag='token', is_lemmatized=True, lemma_tag='lemma', is_pos_tagged=True, pos_tag_tag='pos', **args):
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
        return tag.text


class HistoricalCorpus(SortedKeyList):

    def __new__(cls, *args, **kwargs):
        """Ensures only valid arguments go to SortedKeyList"""
        return super().__new__(cls)

    def __init__(self, corpora:Union[List[Corpus],str], key=lambda c : c.time, corpus_type=None, time_function=None):
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
                if corpus_type not in ['line_by_line','vertical','xml','sprakbanken']:
                    logging.error("When initialising from a folder path, corpus_type must be one of 'line_by_line','vertical','xml' and 'sprakbanken'.")
                    raise ValueError
                corpora_list = []
                for file in os.listdir(corpora):
                    try:
                        if corpus_type == 'line_by_line':
                            corpus = LinebyLineCorpus(os.path.join(corpora,file),time_function=time_function)
                        elif corpus_type == 'vertical':
                            corpus = VerticalCorpus(os.path.join(corpora,file),time_function=time_function)
                        elif corpus_type == 'xml':
                            corpus = XMLCorpus(os.path.join(corpora,file),time_function=time_function)
                        elif corpus_type == 'sprakbanken':
                            corpus = SprakBankenCorpus(os.path.join(corpora,file),time_function=time_function)
                        corpora_list.append(corpus)
                    except: #TODO: proper exception
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

    def search(self, search_terms : List[ str | Pattern | SearchTerm ], index_by_corpus=False):
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
            usages = {} #TODO: make this saveable
            for corpus in self:
                try:
                    usage_dict : UsageDictionary = corpus.search(search_terms)
                except:
                    logging.error(f"Could not search through {corpus.name}.")
                    continue
                for key in usage_dict:
                    if not key in usages:
                        usages[key] = {corpus.name : TargetUsageList()}
                    usages[key][corpus.name] = usage_dict[key]

        else:
            usages = UsageDictionary()
            for corpus in self:
                try:
                    usage_dict : UsageDictionary = corpus.search(search_terms)
                except:
                    logging.error(f"Could not search through {corpus.name}.")
                    continue
                for key in usage_dict:
                    if not key in usages:
                        usages[key] = TargetUsageList()
                    usages[key].extend(usage_dict[key])

        return usages