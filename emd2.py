import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
import faiss
import json
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Initialize SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Path to your folder containing text files
folder_path = "books"
folder_path2 = "books2"
docs = []

# Process each text file in the folder

def read_files_from_folder(folder_path):
    file_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                chunks = text_splitter.split_text(text)

                for i in range(len(chunks)):
                  file_data.append({"file_name": filename, "page_content": chunks[i]})


    return file_data


file_data = read_files_from_folder(folder_path)
file_data2 = read_files_from_folder(folder_path2)


documents = [doc['page_content'] for doc in file_data]
documents2 = [doc['page_content'] for doc in file_data2]
# print(documents[0])

model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

embeddings = model.encode(documents)
embeddings2 = model.encode(documents2)

faiss.normalize_L2(embeddings)
faiss.normalize_L2(embeddings2)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) 
index = faiss.IndexIDMap(index)

dimension2 = embeddings2.shape[1]
index2 = faiss.IndexFlatL2(dimension2) 
index2 = faiss.IndexIDMap(index2)


ids = np.array(range(len(documents)))
ids2 = np.array(range(len(documents2)))

index.add_with_ids(embeddings, ids)
index2.add_with_ids(embeddings2, ids2)

faiss.write_index(index, 'faiss_index.bin')
faiss.write_index(index2, 'faiss_index2.bin')

with open('document_metadata.json', 'w') as f:
    json.dump(file_data, f)

with open('document_metadata2.json', 'w') as f:
    json.dump(file_data2, f)
