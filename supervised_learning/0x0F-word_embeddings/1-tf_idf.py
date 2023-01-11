#!/usr/bin/env python3
"""creates a TF-IDF embedding:"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding:"""
    if vocab is None:
        vocab = []
        vectorizer = TfidfVectorizer()
    else:
        vocab = vocab
        vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = list(vectorizer.get_feature_names())

    return embeddings, features
