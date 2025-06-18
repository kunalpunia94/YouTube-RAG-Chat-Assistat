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
            # Handle full YouTube URL
            parsed_url = urlparse(input_text)

            if "youtube" in parsed_url.netloc:
                return parse_qs(parsed_url.query)["v"][0]
            elif "youtu.be" in parsed_url.netloc:
                return parsed_url.path.lstrip('/')
            else:
                return input_text  # Assume it's already an ID
        except:
            return input_text  # Fallback

    video_id = extract_video_id(video_url)
    st.video(f"https://www.youtube.com/watch?v={video_id}")

with col2:
    question = st.text_input("Ask a question about the video", value="Can you summarize the video briefly?")
    if st.button("Get Answer"):

        # Step 1a - Get Transcript
        # try:
        #     transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        #     transcript = " ".join(chunk["text"] for chunk in transcript_list)
        #     st.success("Transcript loaded successfully.")
        # except TranscriptsDisabled:
        #     st.error("Captions are disabled for this video.")
        #     st.stop()
        # except Exception as e:
        #     st.error(f"Error fetching transcript: {str(e)}")
        #     st.stop()
        try:
            # List all available transcripts for the video
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)

            # Merge manually created + generated transcript objects
            available_transcripts = list(transcript_list_obj._manually_created_transcripts.values()) + \
                                    list(transcript_list_obj._generated_transcripts.values())

            if not available_transcripts:
                st.error("No transcripts found in any language.")
                st.stop()

            # Create label-to-transcript-object mapping
            transcript_map = {
                f"{t.language_code} - {t.language} ({'Auto' if t.is_generated else 'Manual'})": t
                for t in available_transcripts
            }

            # Show dropdown
            selected_label = st.selectbox("Choose transcript language", list(transcript_map.keys()))
            selected_transcript = transcript_map[selected_label]

            # Fetch transcript
            transcript_result = selected_transcript.fetch()
            # st.write("Transcript Chunk Sample:", transcript_result[0])
            # st.write("Type of chunk:", type(transcript_result[0]))  


            # Ensure it's a list of dictionaries before extracting
            # No need for type checks; just get .text attribute directly
            try:
                # Try .text attribute access
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

        

        # Step 1b - Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # Step 1c - Embeddings + Vector Store
        embedding = HuggingFaceEmbeddings(
            # model_name='sentence-transformers/all-MiniLM-L6-v2',
            # model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            model_name='sentence-transformers/distiluse-base-multilingual-cased-v2'
            # model_kwargs={"device": "cpu"}
        )
        vector_store = FAISS.from_documents(chunks, embedding)
        retriever = vector_store.as_retriever(
            search_type="similarity",  # ← Use similarity instead of MMR
            search_kwargs={"k": 5}
        )
        # retriever = vector_store.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k": 10, "fetch_k": 20}
        # )

        # Step 2 - Prompt Setup for RAG
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

        # Step 3 - Format Chunks
        def format_docs(docs, token_limit=1500):
            enc = tiktoken.get_encoding("cl100k_base")  # works well with GPT models
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

        # Step 4 - Check for summarization vs Q&A
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
            relevant_docs = retriever.get_relevant_documents(question)

            with st.spinner("Generating answer using retrieved chunks..."):
                result = main_chain.invoke(question)

            st.markdown("### 💡 Answer")
            if "i don't know" in result.lower() or "context insufficient" in result.lower():
                st.error("⚠️ LLM could not find a confident answer from the retrieved chunks.")
            else:
                st.write(result)


            st.markdown("### 📚 Source Chunks Used")
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**[Source {i+1}]:** {doc.page_content}")
