from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load the pre-trained SBERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

class SentenceInput(BaseModel):
    sentences: list[str]

class SentenceOutput(BaseModel):
    sentence: str
    embedding: list[float]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/hello")
def read_root():
    return {"Hello": "Hello"}

@app.post("/embed")
async def embed_sentences(input_data: SentenceInput):
    sentences = input_data.sentences
    embeddings = model.encode(sentences)

    output_data = []
    for sentence, embedding in zip(sentences, embeddings):
        output = SentenceOutput(sentence=sentence, embedding=embedding.tolist())
        output_data.append(output)

    return output_data

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080, reload=True)
