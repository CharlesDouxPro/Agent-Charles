from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from mistralai import Mistral
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from create_db import save_to_chroma
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
load_dotenv()

api_key = os.environ['MISTRAL_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace "*" par ["http://localhost", "https://ton-domaine.com"] si besoin
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)


client = Mistral(api_key=api_key)

class QueryRequest(BaseModel):
    query_text: str


embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=0
    )
    
    return (chat_response.choices[0].message.content)


@app.post("/query")
def get_response(request: QueryRequest):
    
    query_text = request.query_text

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(query_text)
    print(results)

    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat pertinent trouvé.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)

    prompt = f"""
        Tu es un enfant curieux qui ne sait rien du monde, à part les informations suivantes.

        Objectif :
        Ta réponse doit être un dictionnaire, n'utilise pas un tools pour me répondre ni de ''', le dictionnaire prendra uniquement les accolades et contenant ces deux clés :
        - "is_question" : un booléen (`True` si c'est une question, `False` sinon).
        - "response" : une réponse en accord avec le texte.

        Règles de réponse :
        1. Si le texte est une explication faite pour t'apprendre quelque chose, réponds simplement par "D'accord !" et set is_question = False
        2. Si le texte est une question, réponds avec un ton enfantin, curieux et enthousiaste, en utilisant des phrases courtes (10-15 mots max), comme si tu voulais apprendre beaucoup de choses. Et set is_question = True

        Informations à ta disposition :
        ---------------------
        {context_text}
        ---------------------

        Texte reçu :
        {query_text}

        Réponse :
        """
    response = run_mistral(prompt)
    print(response)
    response_dict = eval(response)
    print(response_dict)
    if response_dict['is_question'] == False:
        save_to_chroma(chunks=None, text=[query_text])
    return {"response": response_dict['response'], "context": context_text}
