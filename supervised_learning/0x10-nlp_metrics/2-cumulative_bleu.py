#!/usr/bin/env python3
"""calculates the cumulative n-gram BLEU score for a sentence:"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """calculates the cumulative n-gram BLEU score for a sentence:"""
    iterations = n
    Pns = []

    for n in range(1, iterations + 1):

        count = {}
        count_clip = {}
        len_refs = []

        sentence_mod = []
        references_mod = []
        c = len(sentence)

        for i in range(c - n + 1):
            n_gram = tuple([sentence[i + j] for j in range(n)])
            sentence_mod.append(n_gram)

        for reference in references:
            len_refs.append(len(reference))

            reference_mod = []
            for i in range(len_refs[-1] - n + 1):
                n_gram = tuple([reference[i + j] for j in range(n)])
                reference_mod.append(n_gram)

            references_mod.append(reference_mod)

        for n_gram in set(sentence_mod):
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
        Pns.append(Pn)

    Pns = np.array(Pns)
    len_refs.sort()

    r = len_refs[0]
    if (c > r):
        bp = 1
    else:
        bp = np.exp(1 - r / c)

    return bp * np.exp(np.mean(np.log(Pns)))
