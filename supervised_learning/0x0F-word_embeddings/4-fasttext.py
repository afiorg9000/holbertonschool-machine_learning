#!/usr/bin/env python3
"""import gensim"""
import gensim


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """gensim"""
    model = gensim.models.FastText(sentences, min_count=min_count,
                                   iter=iterations, size=size,
                                   window=window,
                                   negative=negative,
                                   seed=seed, sg=cbow,
                                   workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=iterations)
    return model
