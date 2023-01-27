#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset():
    """encodes a translation into tokens:"""

    def __init__(self):
        """encodes a translation into tokens:"""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """encodes a translation into tokens:"""
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encodes a translation into tokens:"""
        vocab_size = self.tokenizer_pt.vocab_size
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())
        pt_tokens = np.insert(pt_tokens, 0, vocab_size)
        en_tokens = np.insert(en_tokens, 0, vocab_size)
        pt_tokens = np.append(pt_tokens, vocab_size + 1)
        en_tokens = np.append(en_tokens, vocab_size + 1)

        return pt_tokens, en_tokens
