SelfCheckGPT with LLM Prompting
===================================

- See example usage with OpenAI's GPT API in `openai_gpt3_prompt.py`
- Initial investigation showed that GPT-3 (text-davinci-003) will output either Yes or No 98% of the time, while any remaining outputs can be set to N/A. The output from prompting when comparing the i-th sentence against sample S^n is converted to score x^n_i through the mapping (Yes, 0.0), (No, 1.0), (N/A, 0.5). The final inconsistency score is then calculated using as the average across all samples.
