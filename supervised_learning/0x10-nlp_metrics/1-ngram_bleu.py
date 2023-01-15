#!/usr/bin/env python3
"""calculates the n-gram BLEU score for a sentence:"""


import numpy as np

def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence:"""
    count = {}
    count_clip = {}
    len_refs = []

    sentence_mod = []
    references_mod = []
    c = len(sentence)

    for i in range(c - 1):
        sentence_mod.append((sentence[i], sentence[i + 1]))

    for reference in references:
        len_refs.append(len(reference))

        reference_mod = []
        for j in range(len_refs[-1] - 1):
            reference_mod.append((reference[j], reference[j + 1]))

        references_mod.append(reference_mod)

    for n_gram in list(set(sentence_mod)):
        count[n_gram] = sentence_mod.count(n_gram)

    for reference_mod in references_mod:
        for n_gram in set(reference_mod):
            if n_gram in sentence_mod:
                if n_gram in count_clip.keys():
                    count_clip[n_gram] = max(count_clip[n_gram],
                                             reference_mod.count(n_gram))
                else:
                    count_clip[n_gram] = reference_mod.count(n_gram)

    for n_gram in count.keys():
        if n_gram in count_clip.keys():
            count_clip[n_gram] = min(count[n_gram], count_clip[n_gram])

    Pn = sum(count_clip.values()) / sum(count.values())

    len_refs.sort()

    r = len_refs[0]
    if (c > r):
        bp = 1
    else:
        bp = np.exp(1 - r / c)

    return bp * np.exp(np.log(Pn))
