"""
Example script of using languagechange to compare word representations. 

If used with the command-line 

"""

import argparse
import random

from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import nltk
import torch

from matplotlib import pyplot as plt
from nltk.corpus import stopwords

from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from umap.umap_ import UMAP

from languagechange.benchmark import SemEval2020Task1
from languagechange.models.representation.static import StaticModel, CountModel, PPMI, SVD
from languagechange.corpora import LinebyLineCorpus
from languagechange.models.representation.alignment import OrthogonalProcrustes
from languagechange.models.representation.contextualized import BERT, XL_LEXEME


def corpus_to_static_embeddings(corpus: LinebyLineCorpus, model_type : StaticModel = SVD):
    count_encoder = CountModel(
        corpus, 
        window_size=10, 
        savepath=f'{args.output_folder}/{corpus.name}_count_matrix'
    )
    count_encoder.encode()
    if isinstance(model_type, CountModel):
        count_encoder.load()
        return count_encoder
    
    PPMI_encoder = PPMI(
        count_encoder, 
        shifting_parameter=5, 
        smoothing_parameter=0.75, 
        savepath=f'{args.output_folder}/{corpus.name}_ppmi_matrix'
    )
    PPMI_encoder.encode()
    if isinstance(model_type, PPMI):
        PPMI_encoder.load()
        return PPMI_encoder
    
    svd_encoder = SVD(
        PPMI_encoder, 
        dimensionality=100, 
        gamma=1.0, 
        savepath=f'{args.output_folder}/{corpus.name}_svd_count_matrix'
    )
    svd_encoder.encode()
    svd_encoder.load()
    return svd_encoder


def compare_word_pair(word1 : str, word2 : str, model1: StaticModel, model2: StaticModel):
    word1_model1 = np.asarray(model1[word1].todense())
    word1_model2 = np.asarray(model2[word1].todense())

    word2_model1 = np.asarray(model1[word2].todense())
    word2_model2 = np.asarray(model2[word2].todense())

    print(f'{word1} cosine similarity', cosine_similarity(word1_model1, word1_model2)[0][0])
    print(f'{word2} cosine similarity', cosine_similarity(word2_model1, word2_model2)[0][0])

    return word1_model1, word1_model2, word2_model1, word2_model2


def align_static_embedding(model1: StaticModel, model2: StaticModel) -> tuple[StaticModel, StaticModel]:
    alignment = OrthogonalProcrustes('aligned1','aligned2')
    alignment.align(model1, model2)
    aligned1 = StaticModel('aligned1')
    aligned2 = StaticModel('aligned2')
    aligned1.load()
    aligned2.load()
    return aligned1, aligned2


def most_similar_idxs(word_idx, M, k):
    sims = np.dot(M[word_idx],M.T)
    return np.flip(np.argsort(sims))[:k]

def project_embeddings(embeddings, sentences):
    # Reduce dimensionality to 2D for visualization (using UMAP in this case)
    reducer = UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Extract the 2D x and y coordinates
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]

    # Create scatter plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                               marker=dict(size=10),
                               text=sentences, # Sentences for hover
                               hoverinfo="text"))

    # Customize layout
    fig.update_layout(title='Embeddings of sentences projected using UMAP',
                      xaxis_title='Dimension 1',
                      yaxis_title='Dimension 2',
                      hovermode='closest',# Tooltip shows info of the closest point
                      width=600,
                      height=600,
                      autosize=False
                      )
    # Show plot
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='CompareRepresentations',
                    description='Compare the word representations between two words')
    parser.add_argument('--output_folder', required=False, default='examples/compare-representations/intermediate-outputs')
    args = parser.parse_args()
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    semeval_en = SemEval2020Task1('EN')

    corpus1 = semeval_en.corpus1_token
    corpus2 = semeval_en.corpus2_token

    corpus1 = semeval_en.corpus1_lemma
    corpus2 = semeval_en.corpus2_lemma

    svd_encoder_corpus1 = corpus_to_static_embeddings(corpus1)
    svd_encoder_corpus2 = corpus_to_static_embeddings(corpus2)
    compare_word_pair('plane_nn', 'will', svd_encoder_corpus1, svd_encoder_corpus2)
    aligned1, aligned2 = align_static_embedding(svd_encoder_corpus1, svd_encoder_corpus2)
    compare_word_pair('plane_nn', 'will', aligned1, aligned2)

    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
    
    M = np.asarray(svd_encoder_corpus1.matrix().todense())
    M = preprocessing.normalize(M)
    idx2word = svd_encoder_corpus1.row2word()
    word2idx = {idx2word[i]:i for i in idx2word}

    idx_word = word2idx['plane_nn']
    idxs = most_similar_idxs(idx_word, M, 1000)

    clean_idxs = []
    for i in idxs:
        if idx2word[i].lower() not in stopwords and len(idx2word[i])>3:
            clean_idxs.append(i)

    idxs = clean_idxs[:40]
    vectors = M[idxs]
    words = [idx2word[i] for i in idxs]

    print(words)

    tsne = TSNE(n_components=2)
    vectors_2d = tsne.fit_transform(vectors)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage='average')
    labels = clustering.fit(vectors)

    plt.figure(figsize=(5,5))
    plt.scatter(vectors_2d[:,0], vectors_2d[:,1])
    for i, word in enumerate(words):
        plt.annotate(word, vectors_2d[i])
    plt.show()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    usages = corpus2.search(['bank'])
    random.shuffle(usages['bank'])
    usages['bank'] = usages['bank'][:100]

    bert = BERT('bert-base-uncased',device=device)
    vectors_bert = bert.encode(usages['bank'])
    project_embeddings(vectors_bert, usages['bank'])

    model = XL_LEXEME(device=device)
    vectors_xl_lexeme = model.encode(usages['bank'])
    project_embeddings(vectors_xl_lexeme, usages['bank'])