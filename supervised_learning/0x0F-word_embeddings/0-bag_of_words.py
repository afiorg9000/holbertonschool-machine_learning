#!/usr/bin/env python3
"""creates a bag of words embedding matrix:"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix:"""
    if vocab is None:
        vocab = []
        vectorizer = CountVectorizer()
    else:
        vocab = vocab
        vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = list(vectorizer.get_feature_names())

    return embeddings, features
