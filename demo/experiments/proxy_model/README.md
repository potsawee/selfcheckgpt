Proxy LLM Experiments
===================================

Examples of running LLaMA:

	python llama_logprob_inference.py --llm_model decapoda-research/llama-7b-hf --output_dir llama7b_output_dir

The script ```llama_logprob_inference.py``` will cache token-level log-probabilities (and entropies). Then, you can load the cached results and generate sentence-level or document-level scores.
