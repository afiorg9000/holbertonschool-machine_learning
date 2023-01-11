#!/usr/bin/env python3
"""gensim word2vec"""
import tensorflow.keras as K


def gensim_to_keras(model):
    """the trainable keras Embedding"""
    return K.layers.Embedding(input_dim=model.wv.syn0.shape[0],
                              output_dim=model.wv.syn0.shape[1],
                              weights=[model.wv.syn0],
                              trainable=False)
