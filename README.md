# 📚 Research Paper Guide

An intelligent assistant for understanding research papers through summarization and question-answering using LLMs.

## 🚀 Overview

The **Research Paper Guide** is a two-in-one Streamlit application that:

1. **Summarizes research papers (PDFs)** to generate easy-to-read, structured summaries.
2. **Provides a chatbot** interface to ask context-aware questions about the uploaded paper using retrieval-augmented generation (RAG).

Built using:

* Python 🐍
* Streamlit 🌐
* LangChain 🧠
* Hugging Face Transformers 🤗 / Groq LLMs
* FAISS for vector search

## 📦 Features

* 🧠 Summarization Tool: Upload a PDF and receive a multi-level, detailed summary.
* 💬 Chatbot: Ask questions about the uploaded paper like "What is the proposed method?" or "What are the key findings?"
* ⏳ Real-time feedback during summary generation.
* 📥 Downloadable chat transcript as a PDF.
* 🧠 Contextual memory using vectorstore for paper-specific interactions.

## 🛠️ Installation

```bash
git clone https://github.com/maharshijani05/Research-Paper-Guide.git
cd Research-Paper-Guide
pip install -r requirements.txt
```

## 🖥️ Usage

Run the app locally:

```bash
streamlit run app.py
```

## 📁 Directory Structure

```
├── app.py                  # Streamlit entry point with tabbed UI
├── chatbot.py              # Chatbot interface + vectorstore retrieval
├── summarizer.py           # Summarization pipeline
├── requirements.txt        # Python dependencies
```

## ✨ How It Works

* **Summarization**:

  * Upload PDF → Extract text → Chunk & summarize using LLM → Display summary.
  * Progressive loading messages using `threading` and `st.empty()`

* **Chatbot**:

  * Upload PDF → Vectorize text using FAISS → Ask questions
  * Query is passed to LangChain's QA chain → Response shown in chat

## 🧠 Models Used

* LLM backend: Groq/OpenAI/any HuggingFace model
* Vector embeddings: `sentence-transformers`
* Summarization model: `bart-large-cnn` or similar

## 📄 Example Prompts

* "Summarize the methodology section."
* "What datasets were used?"
* "Explain the experiment results."

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

## 📃 License

[MIT](LICENSE)

---

Made with ❤️ by Maharshi Jani
