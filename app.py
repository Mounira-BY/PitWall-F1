import streamlit as st
import time
from main import initialiser_pitwall

st.set_page_config(page_title="PitWall F1", page_icon="🏎️")

st.title("🏎️ PitWall F1")
st.caption("Ton expert Formule 1 — Système RAG")

# Sidebar
with st.sidebar:
    st.markdown("## 🏎️ PitWall F1")
    st.markdown("*Système RAG — Projet IA Générative*")
    st.divider()

    # Statut
    st.success("🟢 Système actif")
    st.divider()

    # Métriques temps réel — simples
    st.markdown("**Session en cours**")
    
    nb_questions = len([m for m in st.session_state.get("messages", [])
                        if m["role"] == "user"])
    st.metric("Questions posées", nb_questions)

    st.divider()

    # Stack — sobre
    st.markdown("**Stack technique**")
    st.caption("LLM : Mistral via Ollama")
    st.caption("Embeddings : all-MiniLM-L6-v2")
    st.caption("Base vectorielle : ChromaDB")
    st.caption("Reranker : ms-marco-MiniLM-L-6")
    st.caption("Framework : LangChain")

    st.divider()

    # Corpus — sobre
    st.markdown("**Corpus F1**")
    st.caption("15 documents — pilotes, écuries, circuits, technique")

    st.divider()

    if st.button("🗑️ Réinitialiser", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chargement du pipeline
@st.cache_resource
def charger_pipeline():
    return initialiser_pitwall()

chaine = charger_pipeline()

# Historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Message de bienvenue
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Bonjour ! Je suis PitWall 🏎️ Pose-moi une question sur la F1 — pilotes, écuries, circuits, technique...")

# Affichage historique AVEC sources sauvegardées
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Sources visibles dans l'historique
        if msg.get("sources"):
            st.success(f"📚 Sources : {', '.join(msg['sources'])}")
        # Chunks dans l'historique
        if msg.get("chunks"):
            with st.expander("🔍 Voir les chunks RAG utilisés"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i}** — `{chunk['nom']}`")
                    st.text(chunk["contenu"])
                    if i < len(msg["chunks"]):
                        st.divider()

# Saisie utilisateur
question = st.chat_input("Pose ta question F1...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
        "chunks": []
    })
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche dans la base vectorielle..."):
            debut = time.time()
            try:
                reponse = chaine({"question": question})
                texte = reponse["answer"]
                source_docs = reponse.get("source_documents", [])
                sources = list(set([
                    doc.metadata.get("source", "").split("\\")[-1].replace(".txt", "")
                    for doc in source_docs
                ]))
                chunks = [
                    {
                        "nom": doc.metadata.get("source", "").split("\\")[-1].replace(".txt", ""),
                        "contenu": doc.page_content
                    }
                    for doc in source_docs
                ]
            except Exception as e:
                texte = f"Erreur : {str(e)}"
                sources = []
                chunks = []
            duree = time.time() - debut

        st.write(texte)

        # Sources claires (encadré vert visible)
        if sources:
            st.success(f"📚 Sources : {', '.join(sources)}  •  ⏱️ {duree:.1f}s")

        # Chunks RAG
        if chunks:
            with st.expander("🔍 Voir les chunks RAG utilisés"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}** — `{chunk['nom']}`")
                    st.text(chunk["contenu"])
                    if i < len(chunks):
                        st.divider()

    # Sauvegarde COMPLÈTE dans l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": texte,
        "sources": sources,
        "chunks": chunks
    })