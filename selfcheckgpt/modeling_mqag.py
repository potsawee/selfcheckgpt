import re
from typing import Dict, List, Set, Tuple, Union, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice
from selfcheckgpt.utils import prepare_qa_input, prepare_distractor_input, prepare_answering_input
from selfcheckgpt.utils import MQAGConfig, get_prob_distances

# ---------------------------------------------------------------------------------------- #
# Functions for Question Generation & Answering
def question_generation_sentence_level(
    g1_model,
    g1_tokenizer,
    g2_model,
    g2_tokenizer,
    sentence,
    passage,
    num_questions_per_sent,
    device,
):
    qa_input_ids = prepare_qa_input(
            g1_tokenizer,
            context=sentence,
            device=device,
    )
    num_valid_questions = 0
    questions = []
    for q_ in range(num_questions_per_sent):
        # Stage G.1: question+answer generation
        outputs = g1_model.generate(
            qa_input_ids,
            max_new_tokens=128,
            do_sample=True,
        )
        question_answer = g1_tokenizer.decode(outputs[0], skip_special_tokens=False)
        question_answer = question_answer.replace(g1_tokenizer.pad_token, "").replace(g1_tokenizer.eos_token, "")
        question_answer_split = question_answer.split(g1_tokenizer.sep_token)
        if len(question_answer_split) == 2:
            # valid Question + Annswer output
            num_valid_questions += 1
        else:
            continue
        question = question_answer_split[0].strip()
        answer = question_answer_split[1].strip()

        # Stage G.2: Distractor Generation
        distractor_input_ids = prepare_distractor_input(
            g2_tokenizer,
            context = passage,
            question = question,
            answer = answer,
            device = device,
            separator = g2_tokenizer.sep_token,
        )
        outputs = g2_model.generate(
            distractor_input_ids,
            max_new_tokens=128,
            do_sample=True,
        )
        distractors = g2_tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(g2_tokenizer.pad_token, "").replace(g2_tokenizer.eos_token, "")
        distractors = re.sub("<extra\S+>", g2_tokenizer.sep_token, distractors)
        distractors = [y.strip() for y in distractors.split(g2_tokenizer.sep_token)]
        options = [answer] + distractors

        while len(options) < 4:
            # print("Warning: options =", options)
            options.append(options[-1])

        question_item = {
            'question': question,
            'options': options,
        }
        questions.append(question_item)
    return questions


def question_generation_sampling(
    g1_model,
    g1_tokenizer,
    g2_model,
    g2_tokenizer,
    context,
    num_questions,
    device,
):
    qa_input_ids = prepare_qa_input(
            g1_tokenizer,
            context=context,
            device=device,
    )
    max_repeated_sampling = int(num_questions * 1.5) # sometimes generated question+answer is invalid
    num_valid_questions = 0
    questions = []
    for q_ in range(max_repeated_sampling):
        # Stage G.1: question+answer generation
        outputs = g1_model.generate(
            qa_input_ids,
            max_new_tokens=128,
            do_sample=True,
        )
        question_answer = g1_tokenizer.decode(outputs[0], skip_special_tokens=False)
        question_answer = question_answer.replace(g1_tokenizer.pad_token, "").replace(g1_tokenizer.eos_token, "")
        question_answer_split = question_answer.split(g1_tokenizer.sep_token)
        if len(question_answer_split) == 2:
            # valid Question + Annswer output
            num_valid_questions += 1
        else:
            continue
        question = question_answer_split[0].strip()
        answer = question_answer_split[1].strip()

        # Stage G.2: Distractor Generation
        distractor_input_ids = prepare_distractor_input(
            g2_tokenizer,
            context = context,
            question = question,
            answer = answer,
            device = device,
            separator = g2_tokenizer.sep_token,
        )
        outputs = g2_model.generate(
            distractor_input_ids,
            max_new_tokens=128,
            do_sample=True,
        )
        distractors = g2_tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(g2_tokenizer.pad_token, "").replace(g2_tokenizer.eos_token, "")
        distractors = re.sub("<extra\S+>", g2_tokenizer.sep_token, distractors)
        distractors = [y.strip() for y in distractors.split(g2_tokenizer.sep_token)]
        options = [answer] + distractors

        while len(options) < 4:
            options.append(options[-1])

        question_item = {
            'question': question,
            'options': options,
        }
        questions.append(question_item)
        if num_valid_questions == num_questions:
            break
    return questions

def question_generation_beamsearch(
    g1_model,
    g1_tokenizer,
    g2_model,
    g2_tokenizer,
    context,
    num_beams,
    device,
):
    qa_input_ids = prepare_qa_input(
        g1_tokenizer,
        context=context,
        device=device,
    )
    # Stage G.1: question+answer generation
    outputs = g1_model.generate(
        qa_input_ids,
        max_new_tokens=128,
        do_sample=False,
        num_beams=num_beams,
    )
    question_answer = g1_tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(g1_tokenizer.pad_token, "").replace(g1_tokenizer.eos_token, "")
    question_answer_split = question_answer.split(g1_tokenizer.sep_token)
    if len(question_answer_split) == 2:
        question = question_answer_split[0].strip()
        answer = question_answer_split[1].strip()
    else:
        question = question_answer_split[0].strip()
        answer = 'none'

    # Stage G.2: Distractor Generation
    distractor_input_ids = prepare_distractor_input(
        g2_tokenizer,
        context = context,
        question = question,
        answer = answer,
        device = device,
        separator = g2_tokenizer.sep_token,
    )
    outputs = g2_model.generate(
        distractor_input_ids,
        max_new_tokens=128,
        do_sample=False,
        num_beams=num_beams,
    )
    distractors = g2_tokenizer.decode(outputs[0], skip_special_tokens=False)
    distractors = distractors.replace(g2_tokenizer.pad_token, "").replace(g2_tokenizer.eos_token, "")
    distractors = re.sub("<extra\S+>", g2_tokenizer.sep_token, distractors)
    distractors = [y.strip() for y in distractors.split(g2_tokenizer.sep_token)]
    options = [answer] + distractors

    while len(options) < 4:
        options.append(options[-1])

    question_item = {
        'question': question,
        'options': options,
    }
    return [question_item]


def answering(
    a_model,
    a_tokenizer,
    question,
    options,
    context,
    max_seq_length,
    device,
):
    answering_given_passage = prepare_answering_input(
        tokenizer=a_tokenizer,
        question=question,
        options=options,
        context=context,
        device=device,
        max_seq_length=max_seq_length,
    )
    answering_outputs = a_model(**answering_given_passage)
    prob = torch.softmax(answering_outputs['logits'], dim=-1)[0].cpu().numpy()
    return prob

# ---------------------------------------------------------------------------------------- #
# Main MQAG class
class MQAG:
    def __init__(self,
        g1_model_type: str = 'race',
        device = None,
    ):
        assert g1_model_type in ['race', 'squad']
        self.g1_model_type = g1_model_type

        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.inti_generation = False
        self.inti_answering = False
        print(f"MQAG ({g1_model_type}) initialized to {device}")

    def _initialize_generation(self):
        if self.g1_model_type == 'race':
            g1_model_name = MQAGConfig.generation1_race
        elif self.g1_model_type == 'squad':
            g1_model_name = MQAGConfig.generation1_squad

        # Question Generation Systems (G1 & G2)
        self.g1_tokenizer = AutoTokenizer.from_pretrained(g1_model_name)
        self.g1_model = AutoModelForSeq2SeqLM.from_pretrained(g1_model_name)
        self.g2_tokenizer = AutoTokenizer.from_pretrained(MQAGConfig.generation2)
        self.g2_model = AutoModelForSeq2SeqLM.from_pretrained(MQAGConfig.generation2)
        self.g1_model.eval()
        self.g2_model.eval()
        self.g1_model.to(self.device)
        self.g2_model.to(self.device)
        print(f"Initialized Generation")

    def _initialize_answering(self):
        # Question Answering System (A)
        self.a_tokenizer = LongformerTokenizer.from_pretrained(MQAGConfig.answering)
        self.a_model = LongformerForMultipleChoice.from_pretrained(MQAGConfig.answering)
        self.a_model.eval()
        self.a_model.to(self.device)
        print(f"Initialized Answering")

    def score(
        self,
        candidate: str,
        reference: str,
        num_questions: int = 10,
        verbose: bool = False,
    ):
        """
        MQAG score
        :param candidate: text from which questions will be derived, e.g. the summary
        :param reference: text to be used as the ground-truth, e.g. the original document
        :return distances: dict{'kl_div': float, 'counting': float, 'hellinger': float, 'total_variation': float}
        """
        questions = self.generate(context=candidate, do_sample=True, num_questions=num_questions)
        probs_cad = self.answer(questions=questions, context=candidate)
        probs_ref = self.answer(questions=questions, context=reference)
        kl_, ct_, hl_, tv_ = 0, 0, 0, 0
        for i in range(num_questions):
            p1 = probs_cad[i]
            p2 = probs_ref[i]
            kl, ct, hl, tv = get_prob_distances(p1, p2)
            kl_ += kl
            ct_ += ct
            hl_ += hl
            tv_ += tv
        kl_ = kl_ / num_questions
        ct_ = ct_ / num_questions
        hl_ = hl_ / num_questions
        tv_ = tv_ / num_questions
        distances = {'kl_div': kl_, 'counting': ct_, 'hellinger': hl_, 'total_variation': tv_}
        if verbose:
            for i in range(num_questions):
                question, options = questions[i]['question'], questions[i]['options']
                print(f"Q{i+1}: {question}")
                print("(1) [P(.|cand)={:.2f}%]\t[P(.|ref)={:.2f}%]\t{}".format(probs_cad[i][0]*100,  probs_ref[i][0]*100, options[0]))
                print("(2) [P(.|cand)={:.2f}%]\t[P(.|ref)={:.2f}%]\t{}".format(probs_cad[i][1]*100,  probs_ref[i][1]*100, options[1]))
                print("(3) [P(.|cand)={:.2f}%]\t[P(.|ref)={:.2f}%]\t{}".format(probs_cad[i][2]*100,  probs_ref[i][2]*100, options[2]))
                print("(4) [P(.|cand)={:.2f}%]\t[P(.|ref)={:.2f}%]\t{}".format(probs_cad[i][3]*100,  probs_ref[i][3]*100, options[3]))
                print("-------------------------------------------------------------------------------")
        return distances

    @torch.no_grad()
    def generate(
        self,
        context: str,
        do_sample: bool = True,
        num_questions: int = 5,
        **kwargs
    ):
        if self.inti_generation == False:
            self._initialize_generation()
            self.inti_generation = True
        if do_sample:
            questions = question_generation_sampling(
                self.g1_model, self.g1_tokenizer,
                self.g2_model, self.g2_tokenizer,
                context, num_questions, self.device,
            )
        else: # beam_search decoding
            if num_questions != 1:
                print("warning: do_sample is False ---> only 1 sample will be generated")
            if 'num_beams' in kwargs:
                num_beams = kwargs['num_beams']
            else:
                num_beams = 5
            questions = question_generation_beamsearch(
                self.g1_model, self.g1_tokenizer,
                self.g2_model, self.g2_tokenizer,
                context, num_beams, self.device,
            )
        return questions

    @torch.no_grad()
    def answer(
        self,
        questions: List[Dict[str, Any]],
        context: str,
    ):
        """
        :param quetions: List of x where x = {'question': str, 'options': List[str]}
        :param context: string
        :return probs: np.array of dimension (num_questions, 4)
        """
        if self.inti_answering == False:
            self._initialize_answering()
            self.inti_answering = True

        num_questions = len(questions)
        probs = np.zeros((num_questions, 4))
        for i, question_item in enumerate(questions):
            question, options = question_item['question'], question_item['options']
            prob = answering(
                self.a_model, self.a_tokenizer,
                question, options, context,
                max_seq_length=4096, device=self.device,
            )
            probs[i] = prob
        return probs
