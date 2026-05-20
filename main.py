import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# Chargement des documents
def charger_documents():
    loader = DirectoryLoader(
        "documents/",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


# Chunking
def decouper_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)


# Embeddings (HuggingFace)
def creer_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


# Base vectorielle ChromaDB
def creer_base_vectorielle(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="pitwall_f1"
    )

def charger_base_existante(embeddings):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="pitwall_f1"
    )


# Re-ranking : cherche 10 chunks, retrie par pertinence, garde 3
def creer_retriever_avec_reranking(vectorstore):
    retriever_base = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    modele_reranking = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    compresseur = CrossEncoderReranker(model=modele_reranking, top_n=3)
    return ContextualCompressionRetriever(
        base_compressor=compresseur,
        base_retriever=retriever_base
    )


# Prompt Engineering
def creer_prompt():
    template = """Tu es PitWall, un expert passionné de Formule 1.
Tu réponds aux questions en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
Si l'information n'est pas dans le contexte, dis clairement : "Je n'ai pas cette information dans ma base de données."
Réponds toujours en français, de manière claire et précise.
N'invente jamais d'informations.

Contexte F1 :
{context}

Historique de la conversation :
{chat_history}

Question : {question}

Réponse de PitWall :"""

    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )


# Assemblage du pipeline RAG complet
def construire_pipeline(retriever, prompt):
    llm = Ollama(
        model="mistral",
        temperature=0.1,
        base_url="http://localhost:11434"
    )
    memoire = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memoire,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )



def initialiser_pitwall():
    embeddings = creer_embeddings()

    if os.path.exists("./chroma_db"):
        vectorstore = charger_base_existante(embeddings)
    else:
        documents   = charger_documents()
        chunks      = decouper_chunks(documents)
        vectorstore = creer_base_vectorielle(chunks, embeddings)

    retriever = creer_retriever_avec_reranking(vectorstore)
    prompt    = creer_prompt()
    chaine    = construire_pipeline(retriever, prompt)

    return chaine