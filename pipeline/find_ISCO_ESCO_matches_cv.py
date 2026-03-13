import torch
import json 

import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from sentence_transformers import SentenceTransformer


def prepare_iscos(row):
    row["Included occupations"] = row["Included occupations"].replace('Examples of the occupations classified here:', '')
    row["Included occupations"] = "".join(row["Included occupations"].split("\n")[:4]).replace("- ", ", ")

    return {row['ISCO 08 Code']: f"{row['Title EN']}. {row['Definition']} (e.g.{row['Included occupations']})"}

def prepare_escos(row):
    return {f"{row['id']:06d}" : f"{row['preferredLabel']}: {row['description']}"}


def find_matches(query_embedding, ids, matrix_norm, k=20):
    """
    query_embedding: torch.Tensor on cuda, shape [d]
    isco_ids: list of IDs (pre-built)
    isco_matrix_norm: pre-normalized matrix (N, d)
    k: number of matches to return
    """
    # Normalize query
    q = query_embedding / query_embedding.norm(dim=0, keepdim=True)

    # Compute cosine similarity (N)
    sims = torch.matmul(matrix_norm, q)

    # Top-k
    topk_vals, topk_idx = torch.topk(sims, k)

    return [ids[i] for i in topk_idx.tolist()]


def main():
    # Load data
    with open("anon_cvs_coalesced.json", "r") as f:
        cvs = json.load(f)

    candidate_triples = []

    for candidate in cvs:
        current_candidate = []

        cvid = candidate["cvid"]

        if type(candidate["job_history"]) != float:
            # Add candidate jobs
            for job in candidate["job_history"]:
                current_candidate.append((cvid, "has_worked_position", job))

        if type(candidate["jobtitles"]) != float:
            # Add candidate jobtitles
            for job in candidate["jobtitles"]:
                current_candidate.append((cvid, "has_job_title", job))

        if type(candidate["keywords"]) != float:
            # Add candidate keywords
            for kw in candidate["keywords"]:
                current_candidate.append((cvid, "has_keyword", kw))
            
        candidate_triples.append(current_candidate)

    # Load ISCOs
    iscos = pd.read_excel("ISCO-08.xlsx")
    iscos = iscos[(iscos["ISCO 08 Code"].astype(str).str.fullmatch(r'\d{4}'))][["ISCO 08 Code", "Title EN", 
                                                                                "Included occupations", "Definition"]]
    # TODO: Ugly, clean
    result = iscos.apply(lambda row: prepare_iscos(row), axis=1).values

    r_isco = {}

    for d in result:
        r_isco.update(d)

    with open(f"isco_defs.json", "w+") as f:
        f.write(json.dumps(r_isco) + '\n')

    # Load model
    emb_model = SentenceTransformer('dajobbert-kg-specialized')

    isco_embs = {}

    # Compute embedding for both lists
    for isco, text in r_isco.items():
        isco_embs[isco] = emb_model.encode(text, convert_to_tensor=True)    

    isco_ids = list(isco_embs.keys())
    isco_matrix = torch.stack([isco_embs[_id] for _id in isco_ids]).to("cuda:0")  # (N, d)

    # Normalize once for cosine similarity
    isco_matrix_norm = isco_matrix / isco_matrix.norm(dim=1, keepdim=True)

    triples = pd.DataFrame({"candidate": [candidate[0][0] for candidate in candidate_triples], "triples": candidate_triples})

    embedding_list = []
    for candidate in tqdm(candidate_triples, total=len(candidate_triples)):
        if candidate:
            embedding_list.append(emb_model.encode(str(candidate), convert_to_tensor=True).to("cuda:0"))
        else:
            embedding_list.append([])

    triples[f"triples_top_matches_isco"] = [find_matches(emb, isco_ids, isco_matrix_norm, k=20) for emb in embedding_list]

    # Load ESCOs
    df_skills = pd.read_csv("skills_en.csv")
    df_skills["id"] = df_skills.index
    df_skills = df_skills[["id", "skillType", "preferredLabel", "description"]]

    # Prepare ESCOs
    # TODO: Ugly, clean
    result = df_skills.apply(lambda row: prepare_escos(row), axis=1).values

    r_esco = {}

    for d in result:
        r_esco.update(d)

    with open(f"esco_defs.json", "w+") as f:
        f.write(json.dumps(r_esco) + '\n')
    
    embs_esco = {}

    # Compute embeddings for ESCO
    for esco, text in tqdm(r_esco.items()):
        embs_esco[esco] = emb_model.encode(text, convert_to_tensor=True)   

    # Find top matches for ESCO
    esco_ids = list(embs_esco.keys())
    esco_matrix = torch.stack([embs_esco[_id] for _id in esco_ids]).to("cuda:0")  # (N, d)

    # Normalize once for cosine similarity
    esco_matrix_norm = esco_matrix / esco_matrix.norm(dim=1, keepdim=True)

    triples[f"triples_top_matches_esco"] = [find_matches(emb, esco_ids, esco_matrix_norm, k=20) for emb in embedding_list]
   
    triples[["candidate", "triples", "triples_top_matches_isco", "triples_top_matches_esco"]].to_excel("../outputs/clean_outputs/cv_triples_ISCO_ESCO_matches.xlsx")

if __name__ == "__main__":
    main()