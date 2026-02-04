import re
import torch
import json
import random

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, losses, models, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from transformers import BertLMHeadModel

tqdm.pandas()

def extract_triples(input_string):
    """
    Extracts only 3-element tuples, accepting single or double quotes.
    """

    input_string = str(input_string)
    processed_string = input_string.replace('*', '"')

    # Quoting Group (Q): This group (r'["\']') matches either a double quote OR a single quote.
    Q = r'["\']' 
        
    # Flexible Pattern (using f-string for clarity and Q definition):
    pattern = rf"""
        \(              # Match the literal opening parenthesis (
        ({Q}.*?{Q})     # Group 1: Capture the first quoted string
        ,\s* # Match comma, optional whitespace
        ({Q}.*?{Q})     # Group 2: Capture the second quoted string
        ,\s* # Match comma, optional whitespace
        ({Q}.*?{Q})     # Group 3: Capture the third quoted string
        \)              # Match the literal closing parenthesis )
    """
    
    # Use re.findall with re.VERBOSE for multiline pattern and re.DOTALL to match across newlines
    matches = re.findall(pattern, processed_string, re.VERBOSE | re.DOTALL)
    
    extracted_data = []
    for str1_quoted, str2_quoted, str3_quoted in matches:
        # Remove the surrounding quotes from each captured string
        # using the replace method, which handles both ' and "
        str1 = str1_quoted.strip().replace('"', '').replace("'", '')
        str2 = str2_quoted.strip().replace('"', '').replace("'", '')
        str3 = str3_quoted.strip().replace('"', '').replace("'", '')
        extracted_data.append((str1, str2, str3))
        
    return extracted_data

def clean_entity(entity):
    return entity.replace("_", " ").lower()

def extract_entities(triples):
    return list(np.array([[clean_entity(i[0]), clean_entity(i[2])] for i in triples]).flatten().tolist())


def main():
    # Load data
    triples = pd.read_excel("../outputs/clean_outputs/filtered_triples.xlsx").drop("Unnamed: 0", axis=1)

    # Convert to clean triples
    for model in ["qwen", "gemma", "llama"]:
        for prompt in ["structured", "semi-structured", "unstructured"]:
            triples[f"triples_{model}_{prompt}"] = triples[f"triples_{model}_{prompt}"].apply(extract_triples)

    all_entities = []

    # Add subjects and objects to the list of entities
    for model in ["qwen", "gemma", "llama"]:
        for prompt in ["structured", "semi-structured", "unstructured"]:
            all_entities.extend(triples[f"triples_{model}_{prompt}"].progress_apply(extract_entities).values)

    entity_list = list(set([
        entity
        for entities in all_entities
        for entity in entities
    ]))

    # Add CV data too
    with open("../../dataset/final_dataset/anon_cvs.json", 'r', encoding="utf-8") as f:
        data = json.load(f)

    cv_entities = []

    for sample in data:
        for l in ["job_history", "jobtitles", "keywords"]:
            if type(sample[l]) == list:
                cv_entities.extend(sample[l])

    cv_entities = set([clean_entity(i) for i in cv_entities if type(i) == str])  
    entity_list.extend(list(cv_entities))


    ### TRAIN BERT MODEL
    # Setup Device and Data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    unique_entities = list(set(entity_list))
    random.seed(42)
    subset_entities = random.sample(unique_entities, min(100000, len(unique_entities)))
    
    # We provide the sentence twice: once for the noisy input, once for the target.
    train_dataset = Dataset.from_dict({
        "sentence1": subset_entities, 
        "sentence2": subset_entities
    })

    # Setup Model
    word_embedding_model = models.Transformer('jjzha/dajobbert-base-uncased')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Manual Weight Tying 
    decoder = BertLMHeadModel.from_pretrained('jjzha/dajobbert-base-uncased')
    decoder.bert.encoder = model.transformers_model.encoder
    decoder.bert.embeddings = model.transformers_model.embeddings

    # Initialize Loss - tie_encoder_decoder=False because we just did it manually above
    train_loss = losses.DenoisingAutoEncoderLoss(
        model, 
        decoder_name_or_path='jjzha/dajobbert-base-uncased', 
        tie_encoder_decoder=False 
    )
    train_loss.decoder = decoder.to(device)

    # Define Training Arguments
    args = SentenceTransformerTrainingArguments(
        output_dir="dajobbert-checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        fp16=True, 
        save_steps=1000,
        logging_steps=100,
        report_to="none"
    )

    # Initialize Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    # Train and Save
    trainer.train()
    model.save('dajobbert-kg-specialized')
    print("Training complete! Model saved as 'dajobbert-kg-specialized'")


if __name__ == "__main__":
    main()