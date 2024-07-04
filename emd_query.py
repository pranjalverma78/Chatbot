import torch
import os
from sentence_transformers import SentenceTransformer
import re
import faiss
import json
import numpy as np

f_path = 'dataset2.txt'  # Adjust the path if the file is in a different directory

# Read the content of the file and store it in a variable
with open(f_path, 'r') as file:
    datast = file.read()

def create_query_response_list(data):
    lines = data.strip().split('\n')
    query_response_list = []
    current_query = ''

    for line in lines:
        if line.startswith("Query:"):
            current_query = line.replace("Query:", "").strip()
        elif line.startswith("Response:"):
            response = line.replace("Response:", "").strip()
            query_response_list.append(f"Query:{current_query}\nResponse:{response}")
            current_query = ''  # Reset current_query for the next pair

    return query_response_list

documents = create_query_response_list(datast)
# print(type(queries_responses))

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(documents)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) 
index = faiss.IndexIDMap(index)


ids = np.array(range(len(documents)))

index.add_with_ids(embeddings, ids)

faiss.write_index(index, 'faiss_query.bin')

with open('query_metadata.json', 'w') as f:
    json.dump(documents, f)
