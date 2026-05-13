
import streamlit as st
from main import initialiser_pitwall

st.set_page_config(page_title="PitWall F1", page_icon="🏎️")

st.title("🏎️ PitWall F1")
st.caption("Ton expert Formule 1 — RAG + Mistral")

# Chargement du pipeline (une seule fois)
@st.cache_resource
def charger_pipeline():
    return initialiser_pitwall()

chaine = charger_pipeline()

# Historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Saisie utilisateur
question = st.chat_input("Pose ta question F1...")

if question:
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("PitWall réfléchit..."):
            try:
                reponse = chaine({"question": question})
                texte = reponse["answer"]
                sources = [doc.metadata.get("source", "").split("/")[-1].replace(".txt","")
                           for doc in reponse.get("source_documents", [])]
            except Exception as e:
                texte = f"Erreur : {str(e)}"
                sources = []

        st.write(texte)
        if sources:
            st.caption(f"Sources : {', '.join(set(sources))}")

    st.session_state.messages.append({"role": "assistant", "content": texte})