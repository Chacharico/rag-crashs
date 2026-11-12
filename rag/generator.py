import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "chroma_db"

def get_rag_chain(agent_role="general"):
    # 1. Charger la base vectorielle existante
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Définir un prompt dynamique selon l'agent choisi par la Personne B
    system_prompt = (
        "Tu es un assistant spécialisé dans l'analyse de crashs aériens. "
        f"Ton rôle actuel correspond à l'agent : {agent_role}.\n"
        "Utilise les documents contextuels fournis pour répondre à la question.\n"
        "Si tu ne connais pas la réponse, dis clairement que tu ne sais pas.\n\n"
        "Contexte :\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 3. Initialiser le LLM (nécessite une clé API OpenAI dans l'environnement)
    # Tu pourras remplacer par un LLM open-source (ex: Ollama) si besoin
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 4. Créer la chaîne RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain