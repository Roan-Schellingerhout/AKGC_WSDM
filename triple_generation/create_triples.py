import json
import torch
import re
import gc
import getopt
import sys

import pandas as pd

from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer #, Gemma3nForConditionalGeneration

def run_pipeline(model, tokenizer, prompt, text):
    """
    Takes an LLM, tokenizer, prompt, and input text and generates triples
    accordingly. 

    :param model: initialized LLM
    :param tokenizer: initialized tokenizer
    :param prompt: the prompt to generate the triples
    :param text: the cleaned vacancy text
    """
    
    if not text:
        return []

    # Create message
    messages = [
        {"role": "user", "content": prompt + text}
    ]

    # Apply template
    processed_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    model_inputs = tokenizer([processed_text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096 # 16384
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Return response
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def clean_job_blob(text):
    """
    Cleans the input text (the raw vacancies)
    
    :param text: the raw vacancy text
    """


    # Remove <script>...</script> and <style>...</style> blocks
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove @font-face, @media and CSS rules
    text = re.sub(r'@[^{}]+\{[^{}]*\}', '', text, flags=re.DOTALL)
    text = re.sub(r'[.#]?[A-Za-z0-9_\-]+\s*\{[^{}]*\}', '', text, flags=re.DOTALL)

    # Remove anything that looks like JS/JSON objects
    text = re.sub(r'\{[^{}]{50,}\}', '', text, flags=re.DOTALL)  
    text = re.sub(r'\[[^\[\]]{50,}\]', '', text, flags=re.DOTALL)  

    # Remove JS variable or const declarations
    text = re.sub(r'\b(var|const|let)\b[^;{]+[;{]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove escaped unicode (e.g. \u00a9)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)

    # Keep only lines that contain letters
    lines = [l.strip() for l in text.splitlines() if re.search(r'[A-Za-z]', l)]
    text = ' '.join(lines)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_triples(df, model_name, hf, prompt, prompt_type, start, end, device="cpu"):
    """
    Generates the triples for a single model/prompt combination
    
    :param model_name: the short-hand name of the model; [gemma, qwen, llama]
    :param hf: the huggingface identifier of the model
    :param prompt: the prompt being used
    :param prompt_type: the prompt type being used (structured, semi-structured, unstructured)
    :param start: starting index of the current batch
    :param end: ending index of the current batch
    :param device: cpu/cuda
    """

    result = defaultdict(list)

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(hf)

    if model_name == "gemma":
        model = Gemma3nForConditionalGeneration.from_pretrained(
            hf, 
            dtype="auto").to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf,
            dtype="auto"
        ).to(device)

    # Open and load the todo file
    with open("todo.json", "r", encoding="utf-8") as f:
        todo = json.load(f)
    
    for i, row in tqdm(enumerate(df.itertuples()), total=len(df)):
        # Only run for samples in current range
        if not (start <= i <= end):
            continue

        # Indices that have already been completed in a previous run should be skipped
        if not i in todo[model_name][prompt_type]:
            continue

        clean_text = clean_job_blob(row[3])
        result["model"].append(model_name)
        result["prompt"].append(prompt_type)
        result["id"].append(row[1])
        result["text"].append(clean_text)    
        result["triples"].append(run_pipeline(model, tokenizer, prompt, clean_text))

        with open(f"./temporary_results_{model_name}_{prompt_type}.txt") as f:
            f.write(json.dumps(result) + '\n')

    del model
    del tokenizer
    torch.cuda.empty_cache() 
    torch.cuda.ipc_collect()
    gc.collect()

    return result


def main(start, end):
    texts = defaultdict(list)

    with open("../../dataset/final_dataset/jobs.json", 'r') as f:
        data = json.load(f)

    for item in data:
        texts["id"].append(item.get("humanjobid", ""))
        texts["company"].append(item.get("companytext", ""))
        texts["job title"].append(item.get("jobtitle", ""))
        texts["text"].append(item.get("searchtext", ""))

    df = pd.DataFrame(texts)

    # Open and load the prompt file
    with open("prompts.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Access the prompts
    listing_prompts = data["prompts"]["listing"]

    results = defaultdict(list)

    models = {"qwen" : "Qwen/Qwen3-4B-Instruct-2507",
              "gemma" : "google/gemma-3n-e4b-it",
              "llama" : "meta-llama/Llama-3.2-3B-Instruct"}

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running from: {device}")

    # Start the loop
    for model_name, hf in models.items():
        for prompt_type in ["structured", "semi-structured", "unstructured"]:

            print(f"Starting run for model: {model_name}") 
            print(f"  - Prompt type: {prompt_type}")

            prompt = listing_prompts[prompt_type]
            result = get_triples(df, model_name, hf, prompt, prompt_type, start, end, device=device)

            results["model"].extend(result["model"])
            results["prompt"].extend(result["prompt"])
            results["id"].extend(result["id"])
            results["text"].extend(result["text"])    
            results["triples"].extend(result["triples"])

    # In case some model errored, we trim the output so we can properly 
    # create DataFrames
    cutoff = min([len(v) for k, v in results.items()])

    for k, v in results.items():
        results[k] = v[:cutoff]

    # Final results
    df_res = pd.DataFrame(results)
    df_res.to_excel(f"../outputs/raw_outputs/generated_triples_test_{start}_{end}.xlsx")

if __name__ == "__main__":

    args = sys.argv[1:]
    options = "s:e:"
    long_options = ["start=", "end="]

    try:
        arguments, values = getopt.getopt(args, options, long_options)
        for currentArg, currentVal in arguments:
            if currentArg in ("-s", "--start"):
                start = int(currentVal)
            elif currentArg in ("-e", "--end"):
                end = int(currentVal)
    except getopt.error as err:
        print(str(err))

    print(f"Running for index {start} until {end}")

    main(start, end)