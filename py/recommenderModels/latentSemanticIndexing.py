"""
nGramRecommender Model

Author: Tim Dilmaghani

"""

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances, euclidean_distances

#from nltk.tokenize import RegexpTokenizer
#from nltk.stem import LancasterStemmer
#import spacy, pandas as pd
#import numpy as np
#import re
#from stop_words import get_stop_words

def _calculateCosineDistanceMatrix(query_vector, svd_matrix):
    """
    #TODO: DOCSTRING

    """

    return pairwise_distances(query_vector,
                               svd_matrix,
                               metric="cosine",
                               n_jobs=-1)


def _sortScoresAndReturnTopMatches(scores, top_n):
    """
    Returns the top_n results of scores

    Parameters:
    -----------
    scores: list of tuples, positional (e.g. [(0, 0.58), (1, 0.40), (2, 0.21)])
        List of results stored as a tuple where index 0 is the item and index 1 is the score
    top_n: int, positional
        Integer representing the number of scores that should be returned
    """
    return sorted(scores, key=lambda scores: scores[1])[:top_n]


def lsiRecommendatition(user, items, top_n=5, n_components=300, distanceMetric="cosine"):
    """
    Compute query-document similarity scores in a low-rank document representation using SVD

    Parameters:
    -----------
    user: string, positional (example: "a sequence of words")
        Look for recommendation to this user / word
    items: list, positional (example: ["item1", "item2", "item3"])
        Possible recommendatations in the form of a list of strings
    top_n: int, optional (default: 5)
        Number of recommendatitions returned from the model (e.g. 5 = top 5 results) 
    n_components: int, optional (default: 300)
        Number of dimension used in the SVD model as output
    distanceMetric: str, optional (default: "cosine")
        Distance metric used for lsi calculation 
        - 'cosine'
        - 'euclidean'
        TODO...

    """

    # Init TFIDF Vectorizer
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    
    # Init SVD Model

    # TODO: FEATURES MUST BE LOWER THAN N_COMPONENTS 
    # TODO: NEEDs CHECK

    svd_model = TruncatedSVD(n_components=6,
                             algorithm="randomized",
                             n_iter=10,
                             random_state=42)

    # Transformer Pipeline
    svd_transformer = Pipeline([("tfidf", vectorizer),
                                ("svd", svd_model)])

    # Create SVD Matrix
    document_corpus = [item for item in items]
    svd_matrix = svd_transformer.fit_transform(document_corpus)

    # Create query and query vector
    query = [user]
    query_vector = svd_transformer.transform(query)

    # Get cosine distance matrix
    cosineDistanceMatrix = _calculateCosineDistanceMatrix(query_vector, svd_matrix)
    itemIndexAndDistance = [(indexOfItem, distance) for indexOfItem, distance in enumerate(cosineDistanceMatrix[0])]
    
    # Sort and retrieve top n results
    topNitemIndexAndDistance = _sortScoresAndReturnTopMatches(itemIndexAndDistance, top_n)
    
    # Retrieve items by index from corpus
    topNitemAndDistance = [(document_corpus[index], distance) for index, distance in topNitemIndexAndDistance]

    return topNitemAndDistance

user = "Current assets"

items = ["more assets", "lederhose", "eigenkapital", "fremdkapital", "assets under construction", "long-term assets", "current assets", "Current assets", "beer is great", "Feierabend"]

result = lsiRecommendatition(user, items)

print(result)