# 🎬 YouTube RAG Chat Assistant (Multilingual)

An interactive Streamlit application that allows users to ask questions about any YouTube video using its transcript, powered by RAG (Retrieval-Augmented Generation). The app supports **multilingual transcripts** and auto-translates the query to match the transcript language when needed.

---

## 🚀 Features

- 🔍 Ask questions about any YouTube video
- 🧠 Uses RAG: combines embeddings + LLMs for accurate answers
- 🌐 Supports multilingual YouTube transcripts
- 🗣️ Translates user questions to transcript language and answers back in English
- 📝 Auto-summarization of the full video transcript
- 🎥 Built-in video preview (via Streamlit)
- ✅ Works even with auto-generated captions

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenRouter (DeepSeek v3)
- **Embeddings**: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- **Vector Store**: FAISS
- **Transcript Parsing**: `youtube_transcript_api`
- **Translation**: `deep-translator`
- **Prompt Management**: LangChain

---

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/youtube-rag-assistant.git
cd youtube-rag-assistant


2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies:
pip install -r requirements.txt


📄 .env Configuration
Create a .env file in the project root:
OPENROUTER_API_KEY=your_openrouter_api_key_here


▶️ Run the App
streamlit run app.py

🧪 How to Use
1.Paste a YouTube video URL or video ID.
2.Choose your preferred transcript language (auto/manual).
3.Ask a question or type “summarize the video”.
4.Get accurate answers with citations from the transcript!

