import pandas as pd
import numpy as np
import networkx as nx

import torch
import ast
import re
import urllib.parse
import json
import getopt
import time
import sys

from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3nForConditionalGeneration

def clean_label(text):
    # 1. Replace all non-alphanumerics with underscores
    # Note: \w includes letters like Ä, ö, etc.
    s = re.sub(r'[^\w]', '_', str(text)).lower()
    
    # 2. Collapse multiple leading underscores into one
    s = re.sub(r'_+', '_', s)
    
    return urllib.parse.quote(s)


def clean_label(text):
    # 1. Replace all non-alphanumerics with underscores
    # Note: \w includes letters like Ä, ö, etc.
    s = re.sub(r'[^\w]', '_', str(text)).lower()
    
    # 2. Collapse multiple leading underscores into one
    s = re.sub(r'_+', '_', s)
    
    return urllib.parse.quote(s)


def compress_triples(triples, root_prefix="candidate_"):
    """
    Separates the 'Root' (ID) from its 'Attributes' (Skills, Roles, etc.)
    and returns a unique list of strings for the LLM.
    """
    root_id = None
    attributes = set()

    for s, o, _ in triples:
        # Identify the Root ID (usually the one starting with candidate_ or job_)
        if str(s).startswith(root_prefix):
            root_id = s
            attributes.add(o)
        elif str(o).startswith(root_prefix):
            root_id = o
            attributes.add(s)
        else:
            # If neither is the root, they are both descriptive attributes
            attributes.add(s)
            attributes.add(o)

    # Clean up strings (optional: remove underscores for better LLM reasoning)
    clean_attributes = [str(a) for a in attributes if a and str(a).lower() != 'none']
    
    return root_id, clean_attributes


def safe_load_triples(triples_input):
    """
    Safely converts string representation of triples to a list.
    Handles 'nan' values by replacing them with a string or None.
    """
    if pd.isna(triples_input) or triples_input == "":
        return []
    
    # If it's already a list (rare but possible depending on how you loaded the DF)
    if isinstance(triples_input, list):
        return triples_input
    
    try:
        # Replace the literal 'nan' (not in quotes) with 'None' so ast can parse it
        # We use a regex or simple replace if the structure is predictable
        cleaned_input = triples_input.replace(", nan)", ", None)")
        return ast.literal_eval(cleaned_input)
    except Exception as e:
        # Fallback: if it's really messy, use a controlled eval
        try:
            return eval(triples_input, {"__builtins__": {}}, {"nan": np.nan})
        except:
            print(f"Failed to parse: {triples_input[:50]}...")
            return []


def find_bridge(tokenizer, model, cv_triples, vacancy_triples):
    """
    Uses the model (llama/gemma/qwen) to find "bridge triples" between
    the sets of cv triples and vacancy triples. 
    """

    prompt = f"""### ### TASK: CONNECT TWO GRAPHS
You are given a SOURCE graph (Candidate) and a TARGET graph (Vacancy).
Your ONLY job is to create links between them. 

### THE CHALLENGE
Currently, these two graphs have ZERO connections. You must find nodes in the SOURCE that are similar to 
nodes in the TARGET and link them.

### INPUT
SOURCE NODES: {cv_triples}
TARGET NODES: {vacancy_triples}

### INSTRUCTION
Find at ALL pairs of nodes (one from SOURCE, one from TARGET) that are related. 
Return them in this format:
[("Source_Node", "relationship", "Target_Node")]

### CONSTRAINTS
- Every triple MUST contain one node from the SOURCE and one from the TARGET, OR A SHARED NEW NODE (E.G., THEIR COMMON INDUSTRY).
- If you return a triple where both nodes are from the same set, the task is a FAILURE.
- If there are no relevant triples, return []. 
- Use EXACT strings from the lists provided.
- ONLY return the list. 
- NO explanations, NO self-corrections, NO "Step-by-step" reasoning.

### OUTPUT ONLY A RAW PYTHON LIST. NO CODE BLOCKS. NO EXPLANATIONS:
    """
    # Create message
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Apply template
    processed_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    model_inputs = tokenizer([processed_text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096 # 16384
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Return response
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main(df_int, df_cv, df_vacancy, cur_time, model_list=["qwen", "llama", "gemma"], start=0, end=280000):

    models = {"qwen" : "Qwen/Qwen3-4B-Instruct-2507",
              "llama" : "meta-llama/Llama-3.2-3B-Instruct",
              "gemma" : "google/gemma-3n-e4b-it"}

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running from: {device}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Filter run
    models = {k: v for k, v in models.items() if k in model_list}

    final_results = defaultdict(list)

    # Start the loop
    for model_name, hf in models.items():

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
        for prompt_type in ["structured", "semi-structured", "unstructured"]:
            final_results = defaultdict(list)

            for row in tqdm(df_int.itertuples(), total=len(df_int)):
                cv_triples = df_cv[df_cv["id"] == row[3]]["CV_triples"]

                if cv_triples.values:
                    cv_triples = cv_triples.values[0]
                else:
                    continue
                    
                vacancy_triples = df_vacancy[df_vacancy["id"] == row[1]][f"triples_{model_name}_{prompt_type}"]
                
                if vacancy_triples.values:
                    vacancy_triples = vacancy_triples.values[0]
                else:
                    continue

                vacancy_triples = [(clean_label(n1), clean_label(n2), {"relationship": edge}) 
                                for n1, edge, n2 in safe_load_triples(vacancy_triples)]
                cv_triples = [(clean_label(n1), clean_label(n2), {"relationship": edge})
                            for n1, edge, n2 in safe_load_triples(cv_triples)]

                vacancy_heads = list(compress_triples(vacancy_triples))
                vacancy_heads[0] = f"vacancy_{int(row[1])}"
                vacancy_heads = tuple(vacancy_heads)[1]
                    

                bridges = find_bridge(tokenizer, model, compress_triples(cv_triples)[1], vacancy_heads)

                final_results["cvid"].append(row[3])
                final_results["humanjobid"].append(row[1])
                final_results["prompt_type"].append(prompt_type)
                final_results["bridge_triples"].append(bridges)

                with open(f"../outputs/raw_outputs/bridge_triples/bridge_triples_{model_name}_{prompt_type}_{start}_{end}_{cur_time}.json", "w+") as f:
                    f.write(json.dumps(final_results) + '\n')


if __name__ == "__main__":

    df_int = pd.read_csv("../../dataset/final_dataset/contacted_anon.csv")

    # Only keep vacancies with at least 15 interactions
    relevant_vacancies = df_int["cvid"].value_counts()[df_int["cvid"].value_counts() >= 15].index

    # Filter to only relevant vacancies
    df_int = df_int[df_int["cvid"].isin(relevant_vacancies.values)]

    # Load triples
    df_vacancy = pd.read_excel(f"../outputs/clean_outputs/triples_coalesced.xlsx").drop("Unnamed: 0", axis=1)
    df_cv = pd.read_excel("../outputs/clean_outputs/full_cv_triples.xlsx").drop("Unnamed: 0", axis=1)

    args = sys.argv[1:]
    options = "m:s:e:"
    long_options = ["model_list=", "start=", "end="]

    model_list = ["qwen", "llama", "gemma"]
    start = 0
    end = len(df_int)

    try:
        arguments, values = getopt.getopt(args, options, long_options)
        for currentArg, currentVal in arguments:
            if currentArg in ("-m", "--model_list"):
                print(currentVal)
                model_list = eval(currentVal)
            elif currentArg in ("-s", "--start"):
                start = int(currentVal)
            elif currentArg in ("-e", "--end"):
                end = int(currentVal)
    except getopt.error as err:
        print(str(err))

    print(f"Running for {model_list}: index {start} until {end}")

    main(df_int, df_cv, df_vacancy, int(time.time()), model_list, start, end)
