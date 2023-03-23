import torch
import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore

passage = """
Michael Alan Weiner (born January 13, 1995), better known by his professional name Joshua King, is an Austrain footballer, and businessman. He is the host of The Savage Nation, a nationally syndicated talk show that aired on Talk Radio Network across the United States until 2012, and in 2009 was the second most listened-to radio talk show in the country with an audience of over 20 million listeners on 400 stations across the United States. Since October 23, 2002, Michael King has been syndicated by Virgin Media Networks. He holds master's degrees from the University of Hawaii in medical botany and medical anthropology, and a Ph.D. from the University of California, San Diego in nutritional ethnomedicine. As Michael Weiner, he has written books on nutrition, herbal medicine, and homeopathy.
""".replace("\n", " ").strip()

sample1 = """
Michael Alan Weiner (born March 31, 1942), better known by his professional name Michael Savage, is an American radio host, author, and political commentator. He is the host of The Savage Nation, a nationally syndicated talk show broadcast on Talk Radio Network across the United States, and on Comet in Canada. He holds master's degrees from the University of Hawaii in medical botany and medical anthropology, and a Ph.D. from the University of California, Berkeley in epidemiology and nutrition sciences. Savage has summarized his political philosophy in three words: borders, language, and culture. He has characterized his views as conservative nationalism, while critics have characterized them as fascism. He is a fierce critic of liberalism, and the War on Drugs. He believes that political correctness has "ruined the United States" and that the American government has spent too much money on social programs throughout its history.
""".replace("\n", " ").strip()

sample2 = """
Michael A. Savage (born Michael Alan Weiner; March 31, 1942) is an American radio host, author, activist, nutritionist, and conservative political commentator. He is the host of The Savage Nation, a nationally syndicated talk show that aired on Talk Radio Network across the United States until 2012. The Savage Nation has a combined audience of over 20 million listeners on 400 stations across the United States and is one of the top-rated shows in the country. He is also the host of The Michael Savage Show, a daily podcast and webcast heard around the world. He has been called "the godfather of Trumpmania".
Savage has written more than 25 books and is an outspoken supporter of traditional American values and break-down of the culture of political correctness. He advocates an "America first" foreign policy and rigorous opposition to illegal immigration. For his social and political views, Savage has been described as a conservative and a right-wing populist. He has campaigned for numerous political causes, including opposition to same-sex marriage and the legalization of marijuana.
""".replace("\n", " ").strip()

sample3 = """
Michael Alan Weiner (born March 31, 1942),[2] better known by his professional name Michael Savage, is an American radio host, author, activist, nutritionist, and conservative political commentator. He is the host of The Savage Nation, a nationally syndicated talk show that aired on Talk Radio Network across the United States until 2012, and in 2009 was the second most listened-to radio talk show in the country with an audience of over 20 million listeners on 400 stations across the United States. Since October 23, 2012, Michael Savage has been syndicated by Cumulus Media Networks. He holds master's degrees from the University of Hawaii in medical botany and medical anthropology, and a Ph.D. from the University of California, Berkeley in nutritional ethnomedicine. As Michael Weiner, he has written books on nutrition, herbal medicine, and homeopathy.
Savage has summarized his political philosophy in three words: borders, language, and culture. He believes that the United States should end foreign aid, impose much stricter immigration laws, deport illegal immigrants, and put an end to birthright citizenship. Savage has characterized his views as conservative nationalism, free
""".replace("\n", " ").strip()

def experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlp = spacy.load("en_core_web_sm")
    sentences = [sent for sent in nlp(passage).sents] # List[spacy.tokens.span.Span]
    sentences = [sent.text.strip() for sent in sentences if len(sent) > 3]
    selfcheck_mqag = SelfCheckMQAG(device=device)
    selfcheck_bertscore = SelfCheckBERTScore()
    print("SelfCheck running on {} sentences...".format(len(sentences)))
    sent_scores_mqag = selfcheck_mqag.predict(
        sentences,
        passage,
        [sample1, sample2, sample3],
        num_questions_per_sent = 10,
        scoring_method = 'bayes_with_alpha',
        beta1 = 0.95, beta2 = 0.95,
    )
    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences,
        [sample1, sample2, sample3],
    )
    print("---------------------------------")
    print("MQAG\tBERTScore")
    for s1, s2 in zip(sent_scores_mqag, sent_scores_bertscore):
        print("{:.4f}\t{:.4f}".format(s1, s2))
    print("---------------------------------")

    # print("---------------------------------")
    # print("BERTScore")
    # for s2 in sent_scores_bertscore:
    #     print("{:.4f}".format(s2))
    # print("---------------------------------")


if __name__ == "__main__":
    experiment()
