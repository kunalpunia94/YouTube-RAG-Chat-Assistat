# app.py
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
import re
import tiktoken
from urllib.parse import urlparse, parse_qs
from deep_translator import GoogleTranslator
import os

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

# ---- Streamlit UI ----
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")
st.title("🎬 YouTube RAG Chat Assistant")

col1, col2 = st.columns([1, 2])  # Left for video, right for chat


with col1:
    video_url = st.text_input("Enter YouTube Video URL or ID", value="Gfr50f6ZBvo")
    def extract_video_id(input_text):
        try:
            parsed_url = urlparse(input_text)
            if "youtube" in parsed_url.netloc:
                return parse_qs(parsed_url.query)["v"][0]
            elif "youtu.be" in parsed_url.netloc:
                return parsed_url.path.lstrip('/')
            else:
                return input_text
        except:
            return input_text

    video_id = extract_video_id(video_url)
    # Support query parameters from Chrome Extension # added for the youtube extension
    query_params = st.query_params
    if "video_id" in query_params and "question" in query_params:
        video_id = query_params["video_id"][0]
        question = query_params["question"][0]
        auto_trigger = True
    else:
        auto_trigger = False

    st.video(f"https://www.youtube.com/watch?v={video_id}")

with col2:
    question = st.text_input("Ask a question about the video", value="Can you summarize the video briefly?")
    if st.button("Get Answer") or auto_trigger:

        try:
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            available_transcripts = list(transcript_list_obj._manually_created_transcripts.values()) + \
                                    list(transcript_list_obj._generated_transcripts.values())

            if not available_transcripts:
                st.error("No transcripts found in any language.")
                st.stop()

            transcript_map = {
                f"{t.language_code} - {t.language} ({'Auto' if t.is_generated else 'Manual'})": t
                for t in available_transcripts
            }

            selected_label = st.selectbox("Choose transcript language", list(transcript_map.keys()))
            selected_transcript = transcript_map[selected_label]

            # Detect transcript language code
            transcript_language_code = selected_transcript.language_code.lower()

            # Translate question into transcript language
            try:
                question_translated = GoogleTranslator(source='auto', target=transcript_language_code).translate(question)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                question_translated = question

            # Fetch transcript text
            transcript_result = selected_transcript.fetch()
            try:
                transcript = " ".join([
                    chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
                    for chunk in transcript_result
                ])
                st.success(f"Transcript loaded in: {selected_transcript.language}")
            except Exception as e:
                st.error(f"Transcript parsing failed: {str(e)}")
                st.stop()

        except TranscriptsDisabled:
            st.error("Captions are disabled for this video.")
            st.stop()
        except Exception as e:
            st.error(f"Error fetching transcript: {str(e)}")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embedding = HuggingFaceEmbeddings(
            model_name='sentence-transformers/distiluse-base-multilingual-cased-v2'
        )
        vector_store = FAISS.from_documents(chunks, embedding)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

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

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        if "summarize" in question.lower():
            st.info("Detected summary-type question. Using full transcript instead of retrieved chunks.")

            summary_prompt = PromptTemplate(
                template="""
You are a helpful summarization assistant.

Below is the full transcript of a video. Please write a clear and concise summary (3–5 sentences) of what the video is about.

Transcript:
{context}

Summary:
""",
                input_variables=["context"]
            )

            chain = summary_prompt | llm | StrOutputParser()

            with st.spinner("Generating summary from full transcript..."):
                result = chain.invoke({"context": transcript})

            st.markdown("### 📝 Summary")
            if "i don't know" in result.lower() or "insufficient" in result.lower():
                st.error("⚠️ LLM could not generate a confident summary based on the full transcript.")
            else:
                st.write(result)

        else:
            relevant_docs = retriever.get_relevant_documents(question_translated)
            with st.spinner("Generating answer using retrieved chunks..."):
                result = main_chain.invoke(question_translated)

            # Translate answer back to English
            try:
                result = GoogleTranslator(source=transcript_language_code, target='en').translate(result)
            except:
                pass

            st.markdown("### 💡 Answer")
            if "i don't know" in result.lower() or "context insufficient" in result.lower():
                st.error("⚠️ LLM could not find a confident answer from the retrieved chunks.")
            else:
                st.write(result)

            st.markdown("### 📚 Source Chunks Used")
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**[Source {i+1}]:** {doc.page_content}")
