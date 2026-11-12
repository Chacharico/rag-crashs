import os
import shutil
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_DIR = "chroma_db"

def build_vector_store():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("Ancienne base vectorielle supprimée.")
    
    documents = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Le dossier '{DATA_DIR}' vient d'être créé. Ajoute tes PDF dedans.")
        return None

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"Chargé : {file}")

    if not documents:
        print("Aucun PDF trouvé dans le dossier data/.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Nombre total de chunks générés : {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persistent_client = chromadb.PersistentClient(path=DB_DIR)
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=persistent_client,
        collection_name="aviation_accidents"
    )
    
    print("Base vectorielle ChromaDB créée et sauvegardée avec succès.")
    return vector_store

def run_pipeline():
    print("Génération de la base vectorielle en cours...")
    return build_vector_store()

if __name__ == "__main__":
    run_pipeline()