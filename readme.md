# Automated Knowledge Graph Construction for Fair Job Recommendation
## By Roan Schellingerhout, Mesut Kaya, Toine Bogers, Francesco Barile, and Nava Tintarev

### Structure
`/dataloaders`: contains the .pth dataloaders
`/fairness`: contains the fairness evaluation code for all models
`/kg_construction`: contains all the code related to building the full knowledge graph and the candidate-vacancy sub-graphs
`/outputs`: location of all model outputs (all triples)
`/pipeline`: contains the code to perform all the additional steps on triples (bridges, entity resolution, taxonomy linking, etc.)
`/recommendation`: contains the code required for recommendation - dataloader creation and actual models
`/triple_generation`: contains the code to prompt the LLMs to generate triples from the source texts

## Graph overview without bridges

| Model | Prompt | Total Graphs | Valid Paths | Avg Path Len (hops) | Avg Nodes | Avg Edges | Avg Density | Avg Clustering |
|---|---|---|---|---|---|---|---|---|
| qwen | structured | 19129 | 66.04% | 2.90 | 43.2 | 41.6 | 0.0490 | 0.0030 |
| qwen | semi-structured | 19129 | 69.22% | 2.89 | 45.5 | 43.8 | 0.0459 | 0.0028 |
| qwen | unstructured | 19129 | 30.95% | 2.83 | 75.8 | 73.4 | 0.0308 | 0.0003 |
| gemma | structured | 19129 | 44.18% | 2.72 | 43.5 | 40.5 | 0.0491 | 0.0020 |
| gemma | semi-structured | 19129 | 61.70% | 2.86 | 53.0 | 50.4 | 0.0405 | 0.0027 |
| gemma | unstructured | 19129 | 82.37% | 2.98 | 74.7 | 72.9 | 0.0296 | 0.0032 |
| llama | structured | 19129 | 82.51% | 2.99 | 68.2 | 67.2 | 0.0338 | 0.0083 |
| llama | semi-structured | 19129 | 80.17% | 3.04 | 67.5 | 66.5 | 0.0339 | 0.0092 |
| llama | unstructured | 19129 | 23.53% | 3.24 | 74.8 | 73.3 | 0.0310 | 0.0157 |