import torch
import os
from sentence_transformers import SentenceTransformer
import re
import faiss
import json
import numpy as np
# from langchain.vectorstores import Chroma
# from langchain.db import Database
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = "books"
folder_path2 = "books2"

# class FileData:
#     def __init__(self, file_name, page_content):
#         self.metadata = file_name
#         self.page_content = page_content

class CharacterTextSplitter:
    def __init__(self, separator, chunk_size, chunk_overlap, length_function=len, is_separator_regex=False):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.is_separator_regex = is_separator_regex

    def split_text(self, text):
        if self.is_separator_regex:
            segments = re.split(self.separator, text)
        else:
            segments = text.split(self.separator)

        chunks = []
        current_chunk = ""

        for segment in segments:
            if self.length_function(current_chunk + segment) + self.length_function(self.separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = segment + self.separator
            else:
                current_chunk += segment + self.separator

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap
        if self.chunk_overlap > 0:
            overlapping_chunks = []
            for i in range(0, len(chunks)):
                overlapping_chunks.append(chunks[i])
                if i < len(chunks) - 1:
                    overlap = chunks[i][-self.chunk_overlap:]
                    chunks[i+1] = overlap + chunks[i+1]
            chunks = overlapping_chunks

        return chunks

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# chunks = text_splitter.split_text(text)

def read_files_from_folder(folder_path):
    file_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                content = file.read()
                texts = text_splitter.split_text(content)
                # print(len(texts))
                for i in range(len(texts)):
                  if(len(texts[i])<300):
                    continue

                  # file_data.append(FileData(file_name, texts[i]))
                  file_data.append({"file_name": file_name, "page_content": texts[i]})
                # file_data.append({"file_name": file_name, "content": content})

    return file_data

file_data = read_files_from_folder(folder_path)
file_data2 = read_files_from_folder(folder_path2)


documents = [doc['page_content'] for doc in file_data]
documents2 = [doc['page_content'] for doc in file_data2]
print(documents[0])

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
