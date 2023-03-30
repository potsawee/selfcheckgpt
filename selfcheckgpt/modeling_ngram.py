# adapted from https://github.com/seismatica/ngram

import spacy
import numpy as np
from nltk.util import ngrams
from typing import Dict, List, Set, Tuple, Union

class UnigramModelSentenceLevel:
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
        Calculate the average log likelihood of the model on the evaluation text
        """
        # sentences = [sent for sent in self.nlp(text).sents] # List[spacy.tokens.span.Span]
        # sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
        average_logprob = []
        lowest_logprob = []
        logprob_doc = []
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
            average_logprob += [np.mean(logprob_sent)]
            lowest_logprob += [np.min(logprob_sent)]
        average_logprob_doc = np.mean(logprob_doc)
        lowest_logprob_doc = np.min(logprob_doc)
        return {
            'sent': {'average_logprob': average_logprob, 'lowest_logprob': lowest_logprob},
            'doc': {'average_logprob': average_logprob_doc, 'lowest_logprob': lowest_logprob_doc},
        }


class UnigramModelTokenLevel:
    def __init__(self, lowercase: bool = True, strip_token: bool = True) -> None:
        self.token_count = 0
        self.counts = {'<unk>': 0}
        self.lowercase = lowercase
        self.strip_token = strip_token

    def add(self, tokens: List[str]) -> None:
        """
        Add/Count number of unigrams in text, one sentence at a time
        """
        if self.strip_token:
            tokens = [token.strip() for token in tokens]
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
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

    def evaluate(self, tokens: List[str]) -> List[Dict]:
        """
        Calculate the average log likelihood of the model on the evaluation text
        :param evaluation_counter: unigram counter for the text on which the model is evaluated on
        :return: list of token-level logprob
        """
        originals = [token for token in tokens]
        if self.strip_token:
            tokens = [token.strip() for token in tokens]
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
        token_count = 0
        predictions = []
        for token0, token in zip(originals, tokens):
            if token not in self.counts:
                token = '<unk>'
            train_prob = self.probs[token]
            logprob = np.log(train_prob)
            item = {
                'token_id': token_count,
                'token': token0, # stored the original token
                'logprob': logprob,
            }
            predictions.append(item)
            token_count += 1
        return predictions
