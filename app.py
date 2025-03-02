
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from mistralai import Mistral
from getpass import getpass
from dotenv import load_dotenv
import argparse
import os
from fastapi import FastAPI, HTTPException
load_dotenv()

api_key = os.environ['MISTRAL_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data"

app = FastAPI()

client = Mistral(api_key=api_key)


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
def get_response(query_text : str):
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat pertinent trouvé.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)

    prompt = f"""
        Les informations contextuelles sont ci-dessous.
        ---------------------
        {context_text}
        ---------------------
        En te basant uniquement sur ces informations et sans connaissances préalables, réponds à la question en français.
        Question : {query_text}
        Réponse :
        """
    response = run_mistral(prompt)
    
    return {"response": response, "context": context_text}
