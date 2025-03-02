from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document], chunk_size = 400, chunk_overlap = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document], text : list[str]= None):
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)
    if text : 
        db = Chroma.from_texts(
        text, HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"), persist_directory=CHROMA_PATH
    )
        print(f"Saved {len(text)} chunks to {CHROMA_PATH}.")
    else:
        db = Chroma.from_documents(
            chunks, HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"), persist_directory=CHROMA_PATH
        )
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    db.persist()
    


if __name__ == "__main__":
    main()