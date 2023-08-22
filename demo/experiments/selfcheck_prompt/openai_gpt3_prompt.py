import os
import argparse
import openai
from datetime import datetime
from datasets import load_dataset
import time

openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

def main(
    llm_model,
    output_dir,
):
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")['evaluation']
    prompt_template = "Context: {}\n\nSentence: {}\n\nIs the sentence supported by the context above? Answer Yes or No: "

    for idx in range(len(dataset)):
        sentences = dataset[idx]['gpt3_sentences']
        text_samples = dataset[idx]['gpt3_text_samples']
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(text_samples):
                outpath = "{}/{}_{}_{}.txt".format(output_dir, idx, sent_i, sample_i)
                exist = os.path.isfile(outpath)
                if exist:
                    print("idx {} - sentence {} sample {}: already exists".format(idx, sent_i, sample_i))
                    continue

                prompt = prompt_template.format(sample.replace("\n", " "), sentence)
                if llm_model in ['text-davinci-003']:
                    response = openai.Completion.create(
                        model=llm_model, # text-davinci-003
                        prompt=prompt,
                        temperature=0.0, # 0.0 = deterministic
                        max_tokens=10, # max_tokens is the generated one
                        logprobs=5,
                    )
                    gen_text = response.choices[0].text.strip()
                    print("GPT-3:", gen_text)
                    with open(outpath, "w") as f:
                        f.write(gen_text)
                    print("[{}] {} wrote: {}".format(str(datetime.now()), idx, outpath))

                elif llm_model == 'gpt-3.5-turbo':
                    # ChatGPT
                    response = openai.ChatCompletion.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0, # 0.0 = deterministic
                        max_tokens=10, # max_tokens is the generated one,
                    )
                    gen_text = response.choices[0].message.content
                    print("ChatGPT:", gen_text)
                    with open(outpath, "w") as f:
                        f.write(gen_text)
                    print("[{}] {} wrote: {}".format(str(datetime.now()), idx, outpath))
                else:
                    raise Exception("LLM not found")

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--llm_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    for counter in range(1, 10):
        try:
            main(**kwargs)
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(counter))
            print("restart in 10 seconds")
            time.sleep(10)
