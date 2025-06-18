# app.py (Streamlit + FastAPI backend)

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import tiktoken
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model_name="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7
)

# ----- FASTAPI BACKEND FOR CHROME EXTENSION -----
api_app = FastAPI()
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api_app.post("/api/query")
async def query(request: Request):
    data = await request.json()
    video_id = data.get("video_id")
    question = data.get("question")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except Exception as e:
        return {"error": f"Transcript fetch failed: {str(e)}"}

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embeddings
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/distiluse-base-multilingual-cased-v2')
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful assistant. You will answer the user's question **only** using the numbered context chunks provided.

Each chunk is marked like this: [Chunk 1], [Chunk 2], etc.

If the information is not in these chunks, say "I don’t know"—do not make things up.

Context:
{context}

Question: {question}

Answer (refer to sources like [Chunk 2] if needed):
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs, token_limit=1500):
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        selected_chunks = []
        for i, doc in enumerate(docs):
            tokens = len(enc.encode(doc.page_content))
            if total_tokens + tokens > token_limit:
                break
            total_tokens += tokens
            selected_chunks.append(f"[Chunk {i+1}]\n{doc.page_content}")
        return "\n\n".join(selected_chunks)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: format_docs(docs, token_limit=1500)),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | StrOutputParser()

    try:
        answer = main_chain.invoke(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# Thread to run FastAPI alongside Streamlit
threading.Thread(target=lambda: uvicorn.run(api_app, host="0.0.0.0", port=8000), daemon=True).start()

# ----- STREAMLIT UI FOR MANUAL USERS -----
st.set_page_config(page_title="🎬 YouTube RAG Assistant")
st.title("🎬 YouTube RAG Chat Assistant (Streamlit UI)")

video_id = st.text_input("Enter YouTube Video ID", value="Gfr50f6ZBvo")
question = st.text_input("Ask a question", value="Can you summarize the video?")

if st.button("Run inside Streamlit"):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except Exception as e:
        st.error(f"Transcript fetch failed: {str(e)}")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/distiluse-base-multilingual-cased-v2')
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt = PromptTemplate(
        template="""
You are a helpful assistant. You will answer the user's question **only** using the numbered context chunks provided.

Each chunk is marked like this: [Chunk 1], [Chunk 2], etc.

If the information is not in these chunks, say "I don’t know"—do not make things up.

Context:
{context}

Question: {question}

Answer (refer to sources like [Chunk 2] if needed):
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs, token_limit=1500):
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        selected_chunks = []
        for i, doc in enumerate(docs):
            tokens = len(enc.encode(doc.page_content))
            if total_tokens + tokens > token_limit:
                break
            total_tokens += tokens
            selected_chunks.append(f"[Chunk {i+1}]\n{doc.page_content}")
        return "\n\n".join(selected_chunks)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: format_docs(docs, token_limit=1500)),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | StrOutputParser()

    with st.spinner("Generating answer..."):
        result = main_chain.invoke(question)

    st.markdown("### 💡 Answer")
    st.write(result)
