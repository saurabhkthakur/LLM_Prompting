import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import textwrap

import os
from dotenv import load_dotenv
load_dotenv()

class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]


chromodb_NAME = os.environ['DB_NAME']
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

with open('data.txt','r') as file:
    context_data = file.read()
    
context_data = context_data.split('TITLE:')


def chunk_text(text, chunk_size=700):
    return textwrap.wrap(text, chunk_size)

context_chunks = []
for section in context_data:
    context_chunks.extend(chunk_text(section))

# print(len(context_data),len(context_chunks))
chroma_client = chromadb.Client()

db = chroma_client.get_or_create_collection(name=chromodb_NAME, embedding_function=embed_fn)

db.add(documents=context_chunks, ids=[str(i) for i in range(len(context_chunks))])
