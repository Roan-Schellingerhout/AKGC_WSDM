import torch
import gc 
import getopt
import sys
import json
import time

import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
tqdm.pandas()


def run_pipeline(model, tokenizer, r_isco, r_esco, row, model_name, prompt_type):
    
    text = row[f"triples_{model_name}_{prompt_type}"]

    matches = {_id: r_isco[str(_id)] for _id in eval(row[f"triples_{model_name}_{prompt_type}_top_matches_isco"])}
    prompt1 = f"""
    Link the subjects and objects in these triples to their respective ISCO-08 codes (where applicable).
    A link takes the shape: (*subject/object*, has_isco, ISCO_XXXX) with XXXX replaced by the appropriate 5-digit code.
    Ensure that it matches this shape exactly. Only ever use has_isco as the predicate. It is your job to return the 
    triples themselves directly. Never return instructions or code on how those triples could be generated; YOU need
    to generate the relevant ISCO-triples and return those; nothing else. Do not use code for that, but reason over
    which links make sense yourself. 

    You are given 20 possible ISCO codes and their definitions. When returning a triple, ensure you only use the code,
    not the definition. Precisely, you need to evaluate the triples you are provided, determine which ones match ISCO
    codes, and then return new triples containing the subject/object of the triples, has_isco, and then the ISCO code
    that matches it. If none of the triples match an isco definition, it is okay to return an empty list. Therefore,
    choose quality over quantity when it comes to matches. No match is better than a farfetched one. ONLY INCLUDE
    TRIPLES ABOUT THE ISCO CODE (i.e., the ones using has_isco and an ISCO code). NOTHING ELSE. NO OTHER RELATIONS. 
    ONLY ISCO. 
    
    Pick from the following ISCO codes/definitions:
    {matches}
    
    return the found links as a list of triples. Do not add any text to your output other than the subjects, 
    predicates, and objects. The output should therefore precisely be in the shape of [("s1", "p1", "o1"), ("s2", "p2", "o2"), ...]
    with the values replaced. 
    
    Here are the triples: 
    """

    matches2 = {_id: r_esco[f"{_id:06}"] for _id in eval(row[f"triples_{model_name}_{prompt_type}_top_matches_esco"])}

    prompt2 = f"""
    Link the subjects and objects in these triples to their respective ESCO skill codes (where applicable).
    A link takes the shape: (*skill/knowledge*, has_esco, ESCO_*code*) with *skill/knowledge* being replace by the 
    appropriate node and *code* replaced by the appropriate code. Only ever use has_esco as the predicate.
    Ensure that it matches this shape exactly. It is your job to return the triples themselves directly. Never return 
    instructions or code on how those triples could be generated; YOU need to generate the relevant ESCO-triples and
    return those; nothing else. Do not use code for that, but reason over which links make sense yourself. 

    You are given 20 possible ESCO codes and their definitions. When returning a triple, ensure you only use the code,
    not the definition. Precisely, you need to evaluate the triples you are provided, determine which ones match ESCO
    codes, and then return new triples containing the subject/object of the triples, has_isco, and then the ESCO code
    that matches it. If none of the triples match an esco definition, it is okay to return an empty list. Therefore,
    choose quality over quantity when it comes to matches. No match is better than a farfetched one. ONLY INCLUDE
    TRIPLES ABOUT THE ESCO CODES (i.e., ones using has_esco and and ESCO code). NOTHING ELSE. NO OTHER RELATIONS. 
    ONLY ESCO. 
    
    Pick from the following ESCO codes/definitions:
    {matches2}
    
    return the found links as a list of triples. Do not add any text to your output other than the subjects, 
    predicates, and objects. The output should therefore precisely be in the shape of [("s1", "p1", "o1"), ("s2", "p2", "o2"), ...]
    with the values replaced. 

    Here are the triples:
    """

    if not text:
        return []

    output = {}
    
    for t, p in [("isco", prompt1), ("esco", prompt2)]:
        # Create message
        messages = [
            {"role": "user", "content": p + text}
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

        output[t] = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # Return response
    return output


def main(start, end, cur_time):

    print("Loading triples")
    triples = pd.read_excel("../outputs/clean_outputs/triples_ISCO_ESCO_matches.xlsx")

    models = {"qwen" : "Qwen/Qwen3-4B-Instruct-2507",
              "llama" : "meta-llama/Llama-3.2-3B-Instruct",
              "gemma" : "google/gemma-3n-e4b-it"}

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading ISCO and ESCO definitions")    
    # Load ISCO and ESCO definitions
    with open("isco_defs.json", "r") as f:
        r_isco = json.load(f)

    with open("esco_defs.json", "r") as f:
        r_esco = json.load(f)

    with open("todo_ISCO.json", "r") as f:
        todo = json.load(f)

    for model_name, hf in models.items():
        print(f"Loading model: {model_name}")
        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(hf)
        model = AutoModelForCausalLM.from_pretrained(
            hf,
            dtype="auto",
            device_map=None
        ).to(device)

        for prompt in ["structured", "semi-structured", "unstructured"]:
        
            final_results = defaultdict(list)

            for row in tqdm(triples.iterrows(), total=len(triples)):
                
                if not row[0] in todo[model_name][prompt]:
                    continue

                if start < row[0] < end:
                    model_output = run_pipeline(model, tokenizer, r_isco, r_esco, row[1], model_name, prompt)
                    final_results["id"].append(row[0])
                    final_results["model"].append(model_name)
                    final_results["prompt"].append(prompt)
                    final_results["ISCO"].append(model_output["isco"])
                    final_results["ESCO"].append(model_output["esco"])

                    with open(f"./logs_isco_esco/temporary_results_{model_name}_{prompt}_{start}_{end}_{cur_time}.json", "w+") as f:
                       f.write(json.dumps(final_results) + '\n')
           
        del model
        del tokenizer
        torch.cuda.empty_cache() 
        torch.cuda.ipc_collect()
        gc.collect()

        # triples.loc[start:end - 1, f"esco_edges_{model_name}_{prompt}"] = final_results["ESCO"]
        # triples.loc[start:end - 1, f"isco_edges_{model_name}_{prompt}"] = final_results["ISCO"]
            
    
    # triples.to_excel(f"triples_with_ESCO_ISCO_edges_{int(time.time())}.xlsx")

if __name__ == "__main__":

    args = sys.argv[1:]
    options = "s:e:"
    long_options = ["start=", "end="]

    start = 0
    end = 10585

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

    main(start, end, int(time.time()))
