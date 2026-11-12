import streamlit as st
import os
import sys

# Ajouter la racine du projet au path pour importer rag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.generator import get_rag_chain

st.set_page_config(page_title="RAG Crashs Aériens", page_icon="✈️", layout="centered")

st.title("✈️ Assistant RAG — Analyse de Crashs Aériens")
st.markdown("Interroge la base documentaire officielle sur les accidents aériens (AF447, MH370, Überlingen).")

# Sélection fictive ou en attente du module de la Personne B
agent_role = st.selectbox(
    "Agent spécialisé sélectionné (simulé avant intégration de la Personne B) :",
    ["information", "technical", "analysis", "safety"]
)

# Champ de question utilisateur
query = st.text_input("Pose ta question sur les accidents :", placeholder="Ex: Quelles sont les causes techniques de l'accident de...")

if st.button("Rechercher et Générer"):
    if not query.strip():
        st.warning("Veuillez saisir une question.")
    elif not os.path.exists("chroma_db"):
        st.error("La base vectorielle n'a pas été trouvée. Exécute d'abord `python -m rag.pipeline`.")
    else:
        with st.spinner("Recherche et génération de la réponse en cours..."):
            try:
                # Charger la chaîne RAG avec l'agent sélectionné
                chain = get_rag_chain(agent_role=agent_role)
                response = chain.invoke({"input": query})
                
                st.subheader("Réponse :")
                st.write(response["answer"])
                
                # Affichage optionnel des sources (chunks utilisés)
                with st.expander("Voir les sources documentaires utilisées"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source {i+1}** (Fichier : `{os.path.basename(doc.metadata.get('source', 'Inconnue'))}`, Page {doc.metadata.get('page', 'N/A')})")
                        st.text(doc.page_content)
                        st.divider()
                        
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")