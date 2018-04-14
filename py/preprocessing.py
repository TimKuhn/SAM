import numpy as np
import spacy
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

def _loadSpacyModel(lang):
    return spacy.load(lang)

def _porterStemmer(text):
    stemmer = PorterStemmer()
    sents = sent_tokenize(text)
    word_tokens = [" ".join(word_tokenize(sent)) for sent in sents]
    stemmed_word_tokens = [stemmer.stem(word_token) for word_token in word_tokens]
    return " ".join(stemmed_word_tokens)

def _lemmatizer(doc):
    return " ".join([token.lemma_ for token in doc])

def preProcessText(text, language="en", lower=False, removePunctuation=False, lemma=False, stemmer=False, delete_named_entities=False):
                    
    """ Function to pre-process text as specified from user

    TODO: ADD DOCSTRING
    :param text: Takes a string as input
    :param language: 'en' or 'de'
    :param lower:
    :param removePunctuation:
    :param lemma:
    :param stemmer:
    :param delete_named_entities: Removes named entities 

    :returns: 
    
    """
    
    # Look-up table for loading spacy models
    validLanguages = {"en": _loadSpacyModel,
                      "de": _loadSpacyModel}

    if language in validLanguages:
        nlp = validLanguages[language](language)
    else:
        print("{} is not a valid language model".format(language))


    # Lemmatizing
    if lemma:
        doc = nlp(text)
        text = _lemmatizer(doc)

    # Stemmer
    if stemmer:
        text = _porterStemmer(text)

    print(text)

    return text