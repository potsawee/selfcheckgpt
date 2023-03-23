import spacy
import bert_score
import numpy as np
import torch
from typing import Dict, List, Set, Tuple, Union
from transformers import logging
logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice, LongformerForSequenceClassification
from selfcheckgpt.utils import MQAGConfig, expand_list1, expand_list2
from selfcheckgpt.modeling_mqag import question_generation_sentence_level, answering

# ---------------------------------------------------------------------------------------- #
# Functions for counting
def method_simple_counting(
        prob,
        u_score,
        prob_s,
        u_score_s,
        num_samples,
        AT,
):
    """
    simple counting method score => count_mismatch / (count_match + count_mismatch)
    :return score: 'inconsistency' score
    """
    # bad questions, i.e. not answerable given the passage
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_good_sample, count_match = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            count_good_sample += 1
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
    if count_good_sample == 0:
        score = 0.5
    else:
        score = (count_good_sample-count_match) / count_good_sample
    return score

def method_vanilla_bayes(
        prob,
        u_score,
        prob_s,
        u_score_s,
        num_samples,
        beta1, beta2, AT,
):
    """
    (vanilla) bayes method score: compute P(sentence is non-factual | count_match, count_mismatch)
    :return score: 'inconsistency' score
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
            else:
                count_mismatch += 1
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def method_bayes_with_alpha(
        prob,
        u_score,
        prob_s,
        u_score_s,
        num_samples,
        beta1, beta2,
):
    """
    bayes method (with answerability score, i.e. soft-counting) score
    :return score: 'inconsistency' score
    """
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        ans_score = u_score_s[s]
        a_S = np.argmax(prob_s[s])
        if a_DT == a_S:
            count_match += ans_score
        else:
            count_mismatch += ans_score
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def answerability_scoring(
        u_model,
        u_tokenizer,
        question,
        context,
        max_length,
        device,
):
    """
    :return prob: prob -> 0.0 means unanswerable, prob -> 1.0 means answerable
    """
    input_text = question + ' ' + u_tokenizer.sep_token + ' ' + context
    inputs = u_tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    logits = u_model(**inputs).logits
    logits = logits.squeeze(-1)
    prob = torch.sigmoid(logits).item()
    return prob

class SelfCheckMQAG:
    """
    SelfCheckGPT (MQAG varaint): Checking LLM's text against its own sampled texts via MultipleChoice Question Answering
    """
    def __init__(
        self,
        g1_model: str = None,
        g2_model: str = None,
        answering_model: str = None,
        answerability_model: str = None,
        device = None
    ):

        g1_model = g1_model if g1_model is not None else MQAGConfig.generation1_squad
        g2_model = g2_model if g2_model is not None else MQAGConfig.generation2
        answering_model = answering_model if answering_model is not None else MQAGConfig.answering
        answerability_model = answerability_model if answerability_model is not None else MQAGConfig.answerability

        # Question Generation Systems (G1 & G2)
        self.g1_tokenizer = AutoTokenizer.from_pretrained(g1_model)
        self.g1_model = AutoModelForSeq2SeqLM.from_pretrained(g1_model)
        self.g2_tokenizer = AutoTokenizer.from_pretrained(g2_model)
        self.g2_model = AutoModelForSeq2SeqLM.from_pretrained(g2_model)

        # Question Answering System (A)
        self.a_tokenizer = LongformerTokenizer.from_pretrained(answering_model)
        self.a_model = LongformerForMultipleChoice.from_pretrained(answering_model)

        # (Un)Answerability System (U)
        self.u_tokenizer = LongformerTokenizer.from_pretrained(answerability_model)
        self.u_model = LongformerForSequenceClassification.from_pretrained(answerability_model)

        self.g1_model.eval()
        self.g2_model.eval()
        self.a_model.eval()
        self.u_model.eval()

        if device is None:
            device = torch.device("cpu")
        self.g1_model.to(device)
        self.g2_model.to(device)
        self.a_model.to(device)
        self.u_model.to(device)
        self.device = device
        print("SelfCheck-MQAG initialized to device", device)

    @torch.no_grad()
    def predict(
            self,
            sentences: List[str],
            passage: str,
            sampled_passages: List[str],
            num_questions_per_sent: int = 5,
            scoring_method: str = "bayes_with_alpha",
            **kwargs,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param passage: str -- the passage to be evaluated, note that splitting(passage) ---> sentences
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param num_questions_per_sent: int -- number of quetions to be generated per sentence
        :return sent_scores: sentence-level score of the same length as len(sentences) # inconsistency_score, i.e. higher means likely hallucination
        """
        assert scoring_method in ['counting', 'bayes', 'bayes_with_alpha']
        num_samples = len(sampled_passages)
        sent_scores = []
        for sentence in sentences:

            # Question + Choices Generation
            questions = question_generation_sentence_level(
                self.g1_model, self.g1_tokenizer,
                self.g2_model, self.g2_tokenizer,
                sentence, passage, num_questions_per_sent, self.device)

            # Answering
            scores = []
            max_seq_length = 4096 # answering & answerability max length
            for question_item in questions:
                question, options = question_item['question'], question_item['options']
                # response
                prob = answering(
                    self.a_model, self.a_tokenizer,
                    question, options, passage,
                    max_seq_length, self.device)

                u_score = answerability_scoring(
                    self.u_model, self.u_tokenizer,
                    question, passage,
                    max_seq_length, self.device)

                prob_s = np.zeros((num_samples, 4))
                u_score_s = np.zeros((num_samples,))
                for si, sampled_passage in enumerate(sampled_passages):

                    # sample
                    prob_s[si] = answering(
                        self.a_model, self.a_tokenizer,
                        question, options, sampled_passage,
                        max_seq_length, self.device)
                    u_score_s[si] = answerability_scoring(
                        self.u_model, self.u_tokenizer,
                        question, sampled_passage,
                        max_seq_length, self.device)

                # doing comparision
                if scoring_method == 'counting':
                    score = method_simple_counting(prob, u_score, prob_s, u_score_s, num_samples, AT=kwargs['AT'])
                elif scoring_method == 'bayes':
                    score = method_vanilla_bayes(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'], AT=kwargs['AT'])
                elif scoring_method == 'bayes_with_alpha':
                    score = method_bayes_with_alpha(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'])
                scores.append(score)
            sent_score = np.mean(scores)
            sent_scores.append(sent_score)

        return np.array(sent_scores)

class SelfCheckBERTScore:
    """
    SelfCheckGPT (BERTScore variant): Checking LLM's text against its own sampled texts via BERTScore (against best-matched sampled sentence)
    """
    def __init__(self, default_model="en"):
        self.nlp = spacy.load("en_core_web_sm")
        self.default_model = default_model # en => roberta-large
        print("SelfCheck-BERTScore initialized")

    @torch.no_grad()
    def predict(
            self,
            sentences: List[str],
            sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is 1.0 - bertscore
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents] # List[spacy.tokens.span.Span]
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample  = len(sentences_sample)

            refs  = expand_list1(sentences, num_sentences_sample) # r1,r1,r1,....
            cands = expand_list2(sentences_sample, num_sentences) # s1,s2,s3,...

            P, R, F1 = bert_score.score(cands, refs, lang=self.default_model, verbose=False)
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:,s] = F1_arr_max_axis1

        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent
