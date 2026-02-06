import logging
import os
import pickle
from collections import defaultdict
from typing import Tuple, List, Union, Any
import hashlib
from abc import ABC, abstractmethod
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from WordTransformer import WordTransformer, InputExample
from languagechange.usages import TargetUsage
from languagechange.cache import CacheManager

# Configure logging with a basic setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Suppress transformer logging
transformers.logging.set_verbosity_error()

def generate_cache_key(target_usages):
    """
    Generate a unique cache key based on the input data.
    """
    try:
        if isinstance(target_usages, list):
            data = [u.__dict__ if hasattr(u, '__dict__') else u for u in target_usages]
        else:
            data = target_usages.__dict__ if hasattr(target_usages, '__dict__') else target_usages
        serialized = pickle.dumps(data)
        return hashlib.sha256(serialized).hexdigest()
    except Exception as e:
        raise ValueError(f"Invalid input: {e}")

class ContextualizedModel():
    """
    Abstract base class for contextualized embedding models.

    Attributes:
        device (str): The device to run the model on ('cuda' or 'cpu').
        n_extra_tokens (int): Additional tokens to consider during encoding.
    """

    @abstractmethod
    def __init__(self,
                 device: str = 'cuda',
                 n_extra_tokens: int = 0,
                 cache_dir="~/.cache/languagechange/contextualized",
                 *args, **kwargs):
        """
        Initialize the contextualized model.

        Args:
            device (str): 'cuda' or 'cpu'. Defaults to 'cuda'.
            n_extra_tokens (int): Number of extra tokens. Defaults to 0.

        Raises:
            ValueError: If the device is not 'cuda' or 'cpu'.
            ValueError: If n_extra_tokens is not an integer.
        """

        if not device in ['cuda', 'cpu']:
            logger.error("Invalid device specified: Device must be in ['cuda', 'cpu']")
            raise ValueError("Device must be in ['cuda', 'cpu']")
        if not isinstance(n_extra_tokens, int):
            logger.error("n_extra_tokens must be an integer")
            raise ValueError("batch_size must be an integer")

        self._n_extra_tokens = n_extra_tokens
        self._device = device
        self.cache_mgr = CacheManager(cache_dir)

    @abstractmethod
    def encode(self, target_usages: Union[TargetUsage, List[TargetUsage]],
               batch_size: int = 8) -> np.array:
        """
        Encode target usages to generate embeddings.

        Args:
            target_usages (Union[TargetUsage, List[TargetUsage]]): Usage data to encode.
            batch_size (int): Batch size for encoding. Defaults to 8.

        Returns:
            np.array: Encoded embeddings.

        Raises:
            ValueError: If batch_size is not an integer.
            ValueError: If target_usages is not a valid type.
        """

        if not isinstance(batch_size, int):
            logger.error("n_extra_tokens must be an integer")
            raise ValueError("batch_size must be an integer")

        if not (isinstance(target_usages, TargetUsage) or isinstance(target_usages, list)):
            logger.error("target_usages must be Union[dict, List[dict]]")
            raise ValueError("target_usages must be Union[dict, List[dict]]")

class ContextualizedEmbeddings():
    """
    Class to manage contextualized embeddings.
    """
    
    def __str__(self):
        return 'ContextualizedEmbeddings({\n    features: ' + f'{self.column_names}' + f',\n    num_rows: {self.num_rows}' + '\n})'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_usages(target_usages: List[TargetUsage], raw_embedding: np.array):
        columns = defaultdict(list)
        """
        Create a ContextualizedEmbeddings instance from target usages and raw embeddings.

        Args:
            target_usages (List[TargetUsage]): List of target usages.
            raw_embedding (np.array): Embeddings generated from the usages.

        Returns:
            ContextualizedEmbeddings: An instance with formatted data.
        """
        

        for i, target_usage in enumerate(target_usages):
            columns['token'].append(target_usage.token)
            columns['target'].append(target_usage.target)
            columns['context'].append(target_usage.context)
            columns['start'].append(target_usage.start)
            columns['end'].append(target_usage.end)
            columns['embedding'].append(raw_embedding[i])

        embs = ContextualizedEmbeddings.from_dict(columns)
        return embs.with_format("np")

class XL_LEXEME(ContextualizedModel):
    """
    Contextualized model for XL-LEXEME embeddings.
    """

    def __init__(self, pretrained_model: str = 'pierluigic/xl-lexeme',
                 device: str = 'cuda',
                 n_extra_tokens: int = 0):
        """
        Initialize the XL_LEXEME model.

        Args:
            pretrained_model (str): Name of the pretrained model. Defaults to 'pierluigic/xl-lexeme'.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            n_extra_tokens (int): Extra tokens for encoding. Defaults to 0.
        """
        logger.info("Initializing XL_LEXEME model.")
        super().__init__(device=device, n_extra_tokens=n_extra_tokens)

        self._model = WordTransformer(pretrained_model, device=device)

    def encode(self, target_usages: Union[TargetUsage, List[TargetUsage]],
               batch_size: int = 8) -> np.array:
        """ 
        Encode target usages with XL_LEXEME model.

        Args:
            target_usages (Union[TargetUsage, List[TargetUsage]]): Usage data to encode.
            batch_size (int): Batch size for encoding. Defaults to 8.

        Returns:
            np.array: Encoded embeddings.
        """
        
        # Generate cache key
        cache_key = generate_cache_key(target_usages)
        cache_path = os.path.join(self.cache_mgr.cache_dir, f"xl_lexeme_{cache_key}.npy")
        
        # whether the cache files exist
        if os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached embeddings from {cache_path}")
                return np.load(cache_path, allow_pickle=True)
            except Exception as e:
                logger.error(f"Cache loading failed: {str(e)}, deleting corrupted cache file...")
                os.remove(cache_path) 
            
        logger.info("Encoding target usages with batch size: %d", batch_size)
        super(XL_LEXEME, self).encode(target_usages=target_usages, batch_size=batch_size)
        if isinstance(target_usages, TargetUsage):
            target_usages = [target_usages]

        examples = list()

        for target_usage in target_usages:
            start, end = target_usage.offsets
            start, end = int(start), int(end)
            examples.append(InputExample(texts=target_usage.text(), positions=[start, end]))

        raw_embeddings = self._model.encode(examples, batch_size=batch_size, device=self._device)
        
        # save the embedding to file
        with self.cache_mgr.atomic_write(cache_path) as temp_path:
            np.save(temp_path, raw_embeddings)

        return raw_embeddings

class BERT(ContextualizedModel):
    """
    Contextualized model for BERT embeddings.
    """
    
    def __init__(self, pretrained_model: str,
                 device: str = 'cuda',
                 n_extra_tokens: int = 2):
        """
        Initialize the BERT model.

        Args:
            pretrained_model (str): Name of the pretrained model.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            n_extra_tokens (int): Extra tokens for encoding. Defaults to 2.
        """
        
        logger.info("Initializing BERT model.")
        super().__init__(device=device, n_extra_tokens=n_extra_tokens)

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self._model = AutoModel.from_pretrained(pretrained_model)
        self._model.to(device)
        self._token_type_ids = True

    def split_context(self, target_usage: TargetUsage) -> Tuple[List[str], List[str], List[str]]:
        """
        Split the target usage into left, target, and right context tokens.

        Args:
            target_usage (TargetUsage): The usage data.

        Returns:
            Tuple[List[str], List[str], List[str]]: Tokenized left, target, and right context.
        """
        
        logger.info("Splitting context for target usage")
        start, end = target_usage.start(), target_usage.end()

        right_context = target_usage.text()[:start]
        token_occurrence = target_usage.text()[start:end]
        left_context = target_usage.text()[end:]

        left_tokens = self._tokenizer.tokenize(right_context, return_tensors='pt',add_special_tokens=False)
        target_tokens = self._tokenizer.tokenize(token_occurrence, return_tensors='pt',add_special_tokens=False)
        right_tokens = self._tokenizer.tokenize(left_context, return_tensors='pt',add_special_tokens=False)

        return left_tokens, target_tokens, right_tokens

    def center_usage(self, left_tokens: List[str], target_tokens: List[str], right_tokens: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Adjust tokens to fit within the model's maximum sequence length.

        Args:
            left_tokens (List[str]): Tokens from left context.
            target_tokens (List[str]): Tokens from target usage.
            right_tokens (List[str]): Tokens from right context.

        Returns:
            Tuple[List[str], List[str], List[str]]: Trimmed left, target, and right tokens.
        """

        logger.info("Centering usage within maximum sequence length")
        max_seq_len = self._tokenizer.model_max_length
        
        overflow_left = len(left_tokens) - int((max_seq_len -1 -len(target_tokens)) / 2)
        overflow_right = len(right_tokens) - int((max_seq_len -1 -len(target_tokens)) / 2)

        if overflow_left > 0 and overflow_right > 0:
            left_tokens = left_tokens[overflow_left:]
            right_tokens = right_tokens[:len(right_tokens) - overflow_right]

        elif overflow_left > 0 and overflow_right <= 0:
            left_tokens = left_tokens[overflow_left:]

        else:
            right_tokens = right_tokens[:len(right_tokens) - overflow_right]

        return left_tokens, target_tokens, right_tokens

    def add_special_tokens(self, left_tokens: List[str], target_tokens: List[str], right_tokens: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Add special tokens to the tokenized sequences.

        Args:
            left_tokens (List[str]): Left context tokens.
            target_tokens (List[str]): Target tokens.
            right_tokens (List[str]): Right context tokens.

        Returns:
            Tuple[List[str], List[str], List[str]]: Tokenized sequences with special tokens.
        """
        
        logger.info("Adding special tokens")
        left_tokens = [self._tokenizer.cls_token] + left_tokens
        right_tokens = right_tokens + [self._tokenizer.sep_token]
        return left_tokens, target_tokens, right_tokens

    def process_input_tokens(self, tokens: List[str]) -> dict[str, Union[list[int], Any]]:
        """
        Convert tokens to input IDs and attention masks for the model.

        Args:
            tokens (List[str]): Tokens to be processed.

        Returns:
            dict[str, Union[list[int], Any]]: Input IDs, attention masks, and token type IDs.
        """
        
        logger.info("Processing input tokens")
        max_seq_len = self._tokenizer.model_max_length

        input_ids_ = self._tokenizer.convert_tokens_to_ids(tokens[:max_seq_len])
        attention_mask_ = [1] * len(input_ids_)

        offset_len = max_seq_len - len(input_ids_)
        input_ids_ += [self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)] * offset_len
        attention_mask_ += [0] * offset_len

        token_type_ids_ = [0] * len(input_ids_)

        processed_input = {'input_ids': input_ids_,
                           'token_type_ids': token_type_ids_,
                           'attention_mask': attention_mask_}
        if self._token_type_ids:
            del processed_input['token_type_ids']

        return processed_input

    def batch_encode(self, target_usages: List[TargetUsage]) -> np.array:
        """
        Encode a batch of target usages and generate embeddings.

        Args:
            target_usages (List[TargetUsage]): List of target usages.

        Returns:
            np.array: Batch of encoded embeddings.
        """
        
        logger.info("Batch encoding %d target usages", len(target_usages))
        target_embeddings = list()
        examples = defaultdict(list)
        target_offsets = defaultdict(list)

        for target_usage in target_usages:
            left_tokens, target_tokens, right_tokens = self.split_context(target_usage)
            left_tokens, target_tokens, right_tokens = self.center_usage(left_tokens, target_tokens, right_tokens)
            left_tokens, target_tokens, right_tokens = self.add_special_tokens(left_tokens, target_tokens, right_tokens)

            # start and end in terms of tokens
            start, end = len(left_tokens), len(left_tokens) + len(target_tokens)
            target_offsets['start'].append(start)
            target_offsets['end'].append(end)

            tokens = left_tokens + target_tokens + right_tokens
            processed_input = self.process_input_tokens(tokens)

            for k, v in processed_input.items():
                examples[k].append(v)

        for k in examples:
            examples[k] = torch.tensor(examples[k]).to(self._device)

        output = self._model(**examples)

        embeddings = output.last_hidden_state
        for i in range(embeddings.size(0)):
            start, end = target_offsets['start'][i], target_offsets['end'][i]
            target_embedding = embeddings[i, start:end, :].mean(axis=0)
            if self._device == 'cuda':
                target_embeddings.append(target_embedding.detach().cpu().numpy())
            else:
                target_embeddings.append(target_embedding.detach().numpy())

        return np.array(target_embeddings)

    def encode(self, target_usages: Union[TargetUsage, List[TargetUsage]], batch_size: int = 8) -> np.array:
        """
        Encode target usages in batches.

        Args:
            target_usages (Union[TargetUsage, List[TargetUsage]]): List of target usages.
            batch_size (int): Batch size for encoding. Defaults to 8.

        Returns:
            np.array: Array of encoded embeddings.
        """
        
        # Generate cache key
        cache_key = generate_cache_key(target_usages)
        cache_path = os.path.join(self.cache_mgr.cache_dir, f"bert_{cache_key}.npy")
        
        # whether the cache files exist
        if os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached embeddings from {cache_path}")
                return np.load(cache_path, allow_pickle=True)
            except Exception as e:
                logger.error(f"Cache loading failed: {str(e)}, deleting corrupted cache file...")
                os.remove(cache_path) 
            
        logger.info("Starting encoding process with batch size: %d", batch_size)
        super(BERT, self).encode(target_usages=target_usages, batch_size=batch_size)

        target_embeddings = list()

        num_usages = len(target_usages)
        for i in range(0, num_usages, batch_size):
            batch_target_usages = target_usages[i: min(i + batch_size, num_usages)]
            if len(batch_target_usages) > 0:
                target_embeddings.append(self.batch_encode(batch_target_usages))

        raw_embeddings = np.concatenate(target_embeddings, axis=0)

        # save the embedding to file
        with self.cache_mgr.atomic_write(cache_path) as temp_path:
            np.save(temp_path, raw_embeddings)
            
        return raw_embeddings


class RoBERTa(BERT):
    """
    Contextualized model for RoBERTa embeddings, inheriting from BERT.
    """
    
    
    def __init__(self, pretrained_model: str,
                 device: str = 'cuda',
                 n_extra_tokens: int = 2):
        """
        Initialize the RoBERTa model.

        Args:
            pretrained_model (str): Name of the pretrained model.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            n_extra_tokens (int): Extra tokens for encoding. Defaults to 2.
        """
        
        logger.info("Initializing RoBERTa model.")
        super().__init__(pretrained_model=pretrained_model, device=device, n_extra_tokens=n_extra_tokens)

        self._token_type_ids = False
