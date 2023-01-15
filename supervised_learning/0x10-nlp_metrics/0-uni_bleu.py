#!/usr/bin/env python3
"""Calculate the unigram BLEU score."""
from unittest.util import _count_diff_all_purpose
import numpy as np


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score."""
    u_gram = list(set(sentence))

    count_grams = {}

    for reference in references:
        for word in reference:
            if word in u_gram:
                if word not in count_grams:
                    count_grams[word] = reference.count(word)
                else:
                    if reference.count(word) > count_grams[word]:
                        count_grams[word] = reference.count(word)
                    else:
                        pass

    lenght_reference = []
    for reference in references:
        lenght_reference.append(len(reference))

    r = min(lenght_reference)

    c = len(u_gram)

    if c > r:
        bp = 1
    else:
        bp = np.exp(1 - (float(r) / c))

    blue = bp * np.exp(np.log(sum(count_grams.values()) / c))

    return blue
