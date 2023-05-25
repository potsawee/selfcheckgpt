# adapted from https://github.com/seismatica/ngram

import spacy
import numpy as np
from nltk.util import ngrams
from typing import Dict, List, Set, Tuple, Union

class UnigramModel:
    def __init__(self, lowercase: bool = True) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_count = 0
        self.token_count = 0
        self.counts = {'<unk>': 0}
        self.lowercase = lowercase

    def add(self, text: str) -> None:
        """
        Add/Count number of unigrams in text, one sentence at a time
        """
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        for sentence in sentences:
            tokens = [token.text for token in self.nlp(sentence)]
            if self.lowercase:
                tokens = [token.lower() for token in tokens]
            self.sentence_count += 1
            self.token_count += len(tokens)
            for unigram in tokens:
                if unigram not in self.counts:
                    self.counts[unigram] = 1
                else:
                    self.counts[unigram] += 1

    def train(self, k: int = 0) -> None:
        """
        For each unigram in the vocab, calculate its probability in the text
        :param k: smoothing pseudo-count for each unigram
        """
        self.probs = {}
        for unigram, unigram_count in self.counts.items():
            prob_nom = unigram_count + k
            prob_denom = self.token_count + k * len(self.counts) # len(self.counts) = vocab_size
            self.probs[unigram] = prob_nom / prob_denom

    def evaluate(self, sentences: List[str]) -> float:
        """
        Calculate the negative log likelihood of the model on the evaluation sentences
        """
        avg_neg_logprob = []
        max_neg_logprob = []
        logprob_doc = [] # for computing Average at document-level, i.e. Avg(Tokens)
        for sentence in sentences:
            logprob_sent = []
            tokens = [token.text for token in self.nlp(sentence)]
            for token in tokens:
                token_ = token
                if self.lowercase:
                    token = token.lower()
                if token not in self.counts:
                    token_ = '<unk>'
                    token = '<unk>'
                train_prob = self.probs[token]
                logprob = np.log(train_prob)
                logprob_sent.append(logprob)
                logprob_doc.append(logprob)
            avg_neg_logprob += [-1.0 * np.mean(logprob_sent)]
            max_neg_logprob += [-1.0 * np.min(logprob_sent)]
        avg_neg_logprob_doc = -1.0 * np.mean(logprob_doc)
        avg_max_neg_logprob_doc = np.mean(max_neg_logprob)
        return {
            'sent_level': {'avg_neg_logprob': avg_neg_logprob, 'max_neg_logprob': max_neg_logprob},
            'doc_level': {'avg_neg_logprob': avg_neg_logprob_doc, 'avg_max_neg_logprob': avg_max_neg_logprob_doc},
        }

class NgramModel:
    def __init__(
        self, n: int, lowercase: bool = True, left_pad_symbol: str = '<s>') -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_count = 0
        self.ngram_count = 0
        self.counts = {'<unk>': 0}
        self.n = n
        self.lowercase = lowercase
        self.left_pad_symbol = left_pad_symbol

    def add(self, text: str) -> None:
        """
        Add/Count number of ngrams in text, one sentence at a time
        """
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        for sentence in sentences:
            tokens = [token.text for token in self.nlp(sentence)]
            if self.lowercase:
                tokens = [token.lower() for token in tokens]
            ngs = list(ngrams(tokens, n=self.n, pad_left=True, left_pad_symbol=self.left_pad_symbol))
            assert len(ngs) == len(tokens)
            self.sentence_count += 1
            self.ngram_count += len(ngs)
            for ng in ngs:
                if ng not in self.counts:
                    self.counts[ng] = 1
                else:
                    self.counts[ng] += 1

    def train(self, k: int = 0) -> None:
        """
        For each ngram in the vocab, calculate its probability in the text
        :param k: smoothing pseudo-count for each ngram
        """
        self.probs = {}
        for ngram, ngram_count in self.counts.items():
            prob_nom = ngram_count + k
            prob_denom = self.ngram_count + k * len(self.counts) # len(self.counts) = vocab_size
            self.probs[ngram] = prob_nom / prob_denom

    def evaluate(self, sentences: List[str]) -> float:
        """
        Calculate the negative log likelihood of the model on the evaluation sentences
        """
        avg_neg_logprob = []
        max_neg_logprob = []
        logprob_doc = [] # for computing Average at document-level, i.e. Avg(Tokens)
        for sentence in sentences:
            logprob_sent = []
            tokens = [token.text for token in self.nlp(sentence)]
            if self.lowercase:
                tokens_ = [tok.lower() for tok in tokens]
            else:
                tokens_ = [tok for tok in tokens]
            ngs = list(ngrams(tokens_, n=self.n, pad_left=True, left_pad_symbol=self.left_pad_symbol))
            assert len(ngs) == len(tokens)
            for token, ng in zip(tokens, ngs):
                if ng not in self.counts:
                    ng = '<unk>'
                train_prob = self.probs[ng]
                logprob = np.log(train_prob)
                # item = {
                #     'token_id': token_count,
                #     'token': token, # stored the original token
                #     'logprob': logprob,
                # }
                logprob_sent.append(logprob)
                logprob_doc.append(logprob)
            avg_neg_logprob += [-1.0 * np.mean(logprob_sent)]
            max_neg_logprob += [-1.0 * np.min(logprob_sent)]
        avg_neg_logprob_doc = -1.0 * np.mean(logprob_doc)
        avg_max_neg_logprob_doc = np.mean(max_neg_logprob)
        return {
            'sent_level': {'avg_neg_logprob': avg_neg_logprob, 'max_neg_logprob': max_neg_logprob},
            'doc_level': {'avg_neg_logprob': avg_neg_logprob_doc, 'avg_max_neg_logprob': avg_max_neg_logprob_doc},
        }
