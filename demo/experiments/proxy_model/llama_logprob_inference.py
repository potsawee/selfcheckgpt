import os
import argparse
import torch
import pickle
from scipy.stats import entropy
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset

def main(
    llm_model, # e.g, decapoda-research/llama-7b-hf
    output_dir,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained(llm_model)
    model = LlamaForCausalLM.from_pretrained(llm_model)
    model = model.eval()
    model = model.to(device)
    print("loaded:", llm_model)

    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    dataset = dataset['train']
    passages = dataset['gpt3_text']

    for idx, passage in enumerate(passages):
        outpath = "{}/{}.bin".format(output_dir, idx)
        exist = os.path.isfile(outpath)
        if exist:
            print("id {}: already exists".format(idx))
            continue

        passage = passages[idx]
        inputs = tokenizer(passage, return_tensors="pt").to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=-1)[0]
        logprob = torch.log(prob)

        input_ids = inputs.input_ids[0]
        shifted_input_ids = torch.zeros(input_ids.shape, dtype=input_ids.dtype)
        shifted_input_ids[:-1] = input_ids[1:]
        shifted_input_ids[-1] = input_ids[0]


        shifted_input_ids = shifted_input_ids.numpy()
        prob = prob.cpu().numpy()
        ent2 = 2**(entropy(prob, base=2, axis=-1))
        logprob = logprob.cpu().numpy()
        generated_outputs = []
        for t in range(prob.shape[0]):
            gen_tok_id = shifted_input_ids[t]
            gen_tok = tokenizer.decode(gen_tok_id)
            lp = logprob[t, gen_tok_id]
            item = {
                'generation_step': t,
                'gen_tok_id': gen_tok_id,
                'token': gen_tok,
                'logprob': lp,
                'entropy': ent2[t]
            }
            generated_outputs.append(item)

        with open(outpath, 'wb') as f:
            pickle.dump(generated_outputs, f)
        print(f"idx={idx}, outpath={outpath}")

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
    with torch.no_grad():
        main(**kwargs)
