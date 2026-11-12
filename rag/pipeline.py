import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_DIR = "chroma_db"

def build_vector_store():
    # 1. Chargement des PDF du dossier data/
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

    # 2. Chunking (découpage en morceaux)
    # Tu peux tester ici différentes valeurs de chunk_size et chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Nombre total de chunks générés : {len(chunks)}")

    # 3. Embeddings et Base Vectorielle (ChromaDB)
    # Utilisation d'un modèle open-source léger de HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=DB_DIR
    )
    
    print("Base vectorielle ChromaDB créée et sauvegardée avec succès.")
    return vector_store

def search_rag(query, k=3):
    # Charger la base existante
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    results = vector_store.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    # Exécuter la construction de la base au lancement du script
    store = build_vector_store()
    
    if store:
        # Test rapide de recherche
        question = "Quelles sont les causes principales de l'accident ?"
        print(f"\nTest de recherche pour la question : « {question} »\n")
        docs = search_rag(question)
        for i, doc in enumerate(docs):
            print(f"--- Résultat {i+1} (Source : {doc.metadata.get('source', 'Inconnue')}) ---")
            print(doc.page_content[:300] + "...\n")

def run_pipeline():
    # Tout ton code de chargement des PDF, chunking et création de la base ChromaDB
    print("Génération de la base vectorielle en cours...")
    # Exemple :
    # embeddings = HuggingFaceEmbeddings(...)
    # db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

if __name__ == "__main__":
    run_pipeline()
