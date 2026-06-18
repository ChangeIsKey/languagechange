import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from languagechange.usages import TargetUsage
from languagechange.corpora import LinebyLineCorpus
from LSCDetection.modules.utils_ import Space
import os
from collections import defaultdict
import logging
import time
from scipy.sparse import dok_matrix
from gensim.models.word2vec import PathLineSentences
from sklearn.utils.extmath import randomized_svd
from sklearn.random_projection import SparseRandomProjection
from scipy.sparse import csr_matrix
env = os.environ.copy()

# Configure logging with a basic setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepresentationModel(ABC):
    """
    Abstract base class for all representation models. Provides a template for encoding methods.
    """

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        Abstract method for encoding data into a vector representation. Should be implemented by subclasses.
        """
        
        pass

# todo
class StaticModel(RepresentationModel, dict):
    """
    Base class for static word embedding models. Manages loading and accessing vector spaces.
    
    Attributes:
        matrix_path (str): Path to the matrix file.
        format (str): Format of the matrix file (e.g., 'w2v', 'npz').
    """

    def __init__(self, matrix_path=None, format='w2v'):
        logger.info("Initializing StaticModel")
        self.space = None
        self.matrix_path = matrix_path
        self.format = format

    @abstractmethod
    def encode(self):
        """
        Abstract method to perform encoding operations. Must be implemented in subclasses.
        """
        
        pass

    @abstractmethod
    def load(self):
        """
        Load the vector space from the specified file.
        """
        
        logger.info("Loading vector space from %s", self.matrix_path)
        self.space = Space(self.matrix_path, format=self.format)


    def __getitem__(self, k):
        """
        Get the vector representation for a given word.

        Args:
            k (str): The word to look up.
        
        Returns:
            np.array: Vector representation of the word.
        
        Raises:
            Exception: If the space is not loaded.
        """
        
        if self.space == None:
            logger.error('Space is not loaded')
            raise Exception('Space is not loaded')
        return self.space.matrix[self.space.row2id[k]]

    def matrix(self):
        """
        Retrieve the entire matrix of word vectors.

        Returns:
            scipy.sparse.spmatrix: The matrix of word vectors.
        
        Raises:
            Exception: If the space is not loaded.
        """
        
        if self.space == None:
            raise Exception('Space is not loaded')
        return self.space.matrix

    def row2word(self):
        """
        Retrieve the mapping of row indices to words.

        Returns:
            list: List of words corresponding to matrix rows.
        
        Raises:
            Exception: If the space is not loaded.
        """
        
        if self.space == None:
            raise Exception('Space is not loaded')
        return self.space.id2row

class CountModel(StaticModel):
    """
    Count-based word embedding model that builds a co-occurrence matrix from a corpus.

    Attributes:
        corpus (LinebyLineCorpus): The corpus to process.
        window_size (int): The size of the context window.
        savepath (str): Path to save the generated matrix.
    """
    

    def __init__(self, corpus:LinebyLineCorpus, window_size:int, savepath:str):
        super(CountModel,self).__init__()
        self.corpus = corpus
        self.window_size = window_size
        # make sure the path is ending with npz
        if not savepath.endswith('.npz'):
            savepath += '.npz'
        self.savepath = savepath
        self.format = 'npz'
        self.matrix_path = os.path.join(self.savepath)

    def encode(self, is_len = False):
        """
        Build a co-occurrence matrix from the corpus and save it to the specified path.
        """
        
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.count", self.corpus.path, self.savepath, str(self.window_size)])

        # Code below from LSCDetection:
        """
        Make count-based vector space from corpus.
        """ 
        
        # check the cache in the save path
        if os.path.exists(self.savepath):
            try:
                logger.info(f"Loading cached count matrix from {self.savepath}")
                self.load()
                return
            except Exception as e:
                logger.error(f"Cache loading failed: {str(e)}, regenerating matrix...")
                os.remove(self.savepath)
                      
        start_time = time.time()

        # Build vocabulary
        logging.info("Building vocabulary")
        sentences = PathLineSentences(self.corpus.path)
        vocabulary = sorted(list(set([word for sentence in sentences for word in sentence if len(sentence)>1]))) # Skip one-word sentences to avoid zero-vectors
        w2i = {w: i for i, w in enumerate(vocabulary)}
        
        # Initialize co-occurrence matrix as dictionary
        cooc_mat = defaultdict(lambda: 0)

        # Get counts from corpus
        sentences = PathLineSentences(self.corpus.path)
        logging.info("Counting context words")
        for sentence in sentences:
            for i, word in enumerate(sentence):
                lowerWindowSize = max(i-self.window_size, 0)
                upperWindowSize = min(i+self.window_size, len(sentence))
                window = sentence[lowerWindowSize:i] + sentence[i+1:upperWindowSize+1]
                if len(window)==0: # Skip one-word sentences
                    continue
                windex = w2i[word]
                for contextWord in window:
                    cooc_mat[(windex,w2i[contextWord])] += 1

        
        # Convert dictionary to sparse matrix
        logging.info("Converting dictionary to matrix")
        cooc_mat_sparse = dok_matrix((len(vocabulary),len(vocabulary)), dtype=float)
        try:
            cooc_mat_sparse.update(cooc_mat)
        except NotImplementedError:
            for key, value in cooc_mat.items():
                cooc_mat_sparse[key] = value

        outSpace = Space(matrix=cooc_mat_sparse, rows=vocabulary, columns=vocabulary)

        if is_len:
            # L2-normalize vectors
            outSpace.l2_normalize()
            
        # Save the matrix
        outSpace.save(self.savepath)

        logging.info("--- %s seconds ---" % (time.time() - start_time))


class PPMI(CountModel):
    """
    Positive Pointwise Mutual Information (PPMI) model that transforms a co-occurrence matrix.

    Attributes:
        count_model (CountModel): The count-based model to transform.
        shifting_parameter (int): Parameter to shift values after applying log weighting.
        smoothing_parameter (int): Parameter to smooth the matrix values.
        savepath (str): Path to save the PPMI matrix.
    """

    def __init__(self, count_model:CountModel, shifting_parameter:int, smoothing_parameter:int, savepath:str):
        logger.info("Initializing PPMI model")
        super(PPMI,self).__init__(self,count_model.window_size, count_model.savepath)
        self.count_model = count_model
        self.shifting_parameter = shifting_parameter
        self.smoothing_parameter = smoothing_parameter
        self.savepath = savepath
        # self.format ???
        self.matrix_path = os.path.join(self.savepath)
        self.align_strategies = {'OP', 'SRV', 'WI'}

    def encode(self, is_len = False):
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.ppmi", self.count_model.matrix_path, self.savepath, str(self.shifting_parameter), str(self.smoothing_parameter)])

        # Code below from LSCDetection
        """
        Compute the smoothed and shifted PPMI matrix from a co-occurrence matrix. Smoothing is performed as described in

        Omer Levy, Yoav Goldberg, and Ido Dagan. 2015. Improving distributional similarity with lessons learned from word embeddings. Trans. ACL, 3.

        """
        
        # check the cache in savepath
        if os.path.exists(self.savepath):
            try:
                logger.info(f"Loading cached PPMI matrix from {self.savepath}")
                self.load()
                return
            except Exception as e:
                logger.error(f"Cache loading failed: {str(e)}, recomputing PPMI...")
                os.remove(self.savepath)
                

        logger.info("Starting PPMI encoding process")
        start_time = time.time()    

        # Load input matrix
        space = Space(self.count_model.matrix_path)   

        # Apply transformations
        logger.info("Applying transformations: EPMI weighting, log weighting, shifting")
        
        # Apply EPMI weighting
        space.epmi_weighting(self.smoothing_parameter)
        
        # Apply log weighting
        space.log_weighting()

        # Shift values
        space.shifting(self.shifting_parameter)

        # Eliminate negative counts
        space.eliminate_negative()

        # Eliminate zero counts
        space.eliminate_zeros()
            
        outSpace = Space(matrix=space.matrix, rows=space.rows, columns=space.columns)

        if is_len:
            # L2-normalize vectors
            outSpace.l2_normalize()
            
        # Save the matrix
        outSpace.save(self.savepath)

        logger.info("PPMI encoding completed in %s seconds", time.time() - start_time)
        
        
class SVD(StaticModel):
    """
    Singular Value Decomposition (SVD) model that reduces the dimensionality of a matrix.

    Attributes:
        count_model (CountModel): The input count-based model.
        dimensionality (int): Target dimensionality for the reduced matrix.
        gamma (float): Weighting parameter for singular values.
        savepath (str): Path to save the reduced matrix.
    """

    def __init__(self, count_model:CountModel, dimensionality:int, gamma:float, savepath:str):
        logger.info("Initializing SVD model")
        super(SVD,self).__init__()
        self.count_model = count_model
        self.dimensionality = dimensionality
        self.gamma = gamma
        # make sure the save path is end with .w2v
        if not savepath.endswith('.w2v'):
            savepath += '.w2v'
        self.savepath = savepath
        self.matrix_path = os.path.join(self.savepath)
        self.format = 'w2v'
        self.align_strategies = {'OP', 'SRV', 'WI'}

    def encode(self, is_len = False):
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.svd", self.count_model.matrix_path, self.savepath, str(self.dimensionality), str(self.gamma)])

        # Code below from LSCDetection
        """
        Perform dimensionality reduction on a (normally PPMI) matrix by applying truncated SVD as described in

        Omer Levy, Yoav Goldberg, and Ido Dagan. 2015. Improving distributional similarity with lessons learned from word embeddings. Trans. ACL, 3.

        """

        # check the cache file from savepath
        if os.path.exists(self.savepath):
            try:
                logger.info(f"Loading cached SVD matrix from {self.savepath}")
                self.load()
                return
            except Exception as e:
                logger.error(f"Cache loading failed: {str(e)}, recomputing SVD decomposition...")
                os.remove(self.savepath)
                
        logger.info("Starting SVD encoding process")
        start_time = time.time()    

        # Load input matrix
        space = Space(self.count_model.matrix_path)   
        matrix = space.matrix
        
        # Get mappings between rows/columns and words
        rows = space.rows
        id2row = space.id2row
        id2column = space.id2column

        # Apply SVD
        logger.info("Applying truncated SVD")
        u, s, v = randomized_svd(matrix, n_components=self.dimensionality, n_iter=5, transpose=False)

        # Weight matrix
        if self.gamma == 0.0:
            matrix_reduced = u
        elif self.gamma == 1.0:
            #matrix_reduced = np.dot(u, np.diag(s)) # This is equivalent to the below formula (because s is a flattened diagonal matrix)
            matrix_reduced = s * u
        else:
            #matrix_ = np.dot(u, np.power(np.diag(s), gamma)) # This is equivalent to the below formula
            matrix_reduced = np.power(s, self.gamma) * u
        
        outSpace = Space(matrix=matrix_reduced, rows=rows, columns=[])

        if is_len:
            # L2-normalize vectors
            outSpace.l2_normalize()
            
        # Save the matrix
        outSpace.save(self.savepath, format='w2v')

        logger.info("SVD encoding completed in %s seconds", time.time() - start_time)


# todo: add corpus
class RandomIndexing(StaticModel):
    """
    Random Indexing model that creates low-dimensional vector spaces from a co-occurrence matrix.

    Attributes:
        window_size (int): Size of the context window for random indexing.
    """

    def __init__(self):
        logger.info("Initializing RandomIndexing model")
        super(RandomIndexing,self).__init__()
        self.align_strategies = {'OP', 'SRV', 'WI'}
        pass

    def encode(self, is_len = False):
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.ri", corpus.path, self.savepath, self.window_size])

        # Code below from LSCDetection
        """
        Create low-dimensional vector space by sparse random indexing from co-occurrence matrix.
        """        
        
        logger.info("Starting RandomIndexing encoding process")
        start_time = time.time()    
        
        # Load input matrix
        countSpace = Space(corpus.path)   # todo: corpus needs to reference something here
        countMatrix = countSpace.matrix
        rows = countSpace.rows
        columns = countSpace.columns
        
        # Generate random vectors
        #randomMatrix = csr_matrix(sparse_random_matrix(self.window_size,len(columns)).toarray().T)

        #logging.info("Multiplying matrices")
        #reducedMatrix = np.dot(countMatrix,randomMatrix)  
        randomMatrix = SparseRandomProjection(self.window_size).fit_transform(countMatrix)  
        outSpace = Space(matrix=randomMatrix, rows=rows, columns=[])
        
        if is_len:
            # L2-normalize vectors
            outSpace.l2_normalize()

        # Save the matrix
        outSpace.save(self.savepath, format='w2v')

        logger.info("RandomIndexing encoding completed in %s seconds", time.time() - start_time)