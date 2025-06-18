from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

openai_base_url = "https://openrouter.ai/api/v1"
api_key = os.getenv("OPENROUTER_API_KEY")

# this is our model name
llm = ChatOpenAI(
    openai_api_base=openai_base_url,
    openai_api_key=api_key,
    model_name="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7
)


# step 1a - Indexing(Document ingestion)
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])#here we can change language also
except TranscriptsDisabled:
    print("No captions available for this video.")
    transcript_list = []


# Flatten it to plain text(into one )
transcript = " ".join(chunk["text"] for chunk in transcript_list)
# print(transcript) # printed in single



# Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])


# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
vector_store = FAISS.from_documents(chunks, embedding)


# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# Step 3 - Augmentation
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)



# Building the chain
# this function convert all text into together
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke('Can you summarize the video breifly')

print(result)