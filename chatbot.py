import os
from PyPDF2 import PdfReader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import tempfile
from langchain.schema import Document
import streamlit as st

# Load .env once when module is loaded
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY not found in environment variables.")

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Initialize LLM & Embeddings once
llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which may reference context, reformulate it to be self-contained."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that answers questions based on the context of a research paper provided. "
               "If the question is not answerable with the given context, respond with 'I don't know'. "
               "If the question is unrelated, say 'This question is not related to the research paper content.'\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Shared session store for chat history and vectorstores
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = {"history": ChatMessageHistory(), "vectorstore": None}
    return session_store[session_id]["history"]

def get_vectorstore(session_id: str):
    return session_store.get(session_id, {}).get("vectorstore", None)

def set_vectorstore(session_id: str, vectorstore):
    if session_id not in session_store:
        session_store[session_id] = {"history": ChatMessageHistory(), "vectorstore": vectorstore}
    else:
        session_store[session_id]["vectorstore"] = vectorstore

def build_vectorstore_from_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in splits]
    return FAISS.from_documents(docs, embeddings)


def create_chains_for_vectorstore(vectorstore):
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    return conversational_chain

def run_paper_chatbot(user_input: str, session_id: str = "default_session") -> str:
    # Get vectorstore for this session
    vectorstore = get_vectorstore(session_id)
    if vectorstore is None:
        return "Please upload a research paper PDF first."

    conversational_chain = create_chains_for_vectorstore(vectorstore)

    try:
        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response["answer"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app interface for chatbot
def chatbot_app():
    st.title("🤖 Research Paper Chatbot")

    session_id = st.session_state.get("session_id", "default_session")

    # Initialize session variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vectorstore_created" not in st.session_state:
        st.session_state["vectorstore_created"] = False

    # File upload - only show if vectorstore not created
    if not st.session_state["vectorstore_created"]:
        uploaded_file = st.file_uploader("📄 Upload your research paper PDF", type=["pdf"], key="chatbot_uploader")
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                paper_text = extract_text_from_pdf(uploaded_file)
                vectorstore = build_vectorstore_from_text(paper_text)
                set_vectorstore(session_id, vectorstore)
                st.session_state["vectorstore_created"] = True
            st.success("✅ PDF processed! You can now ask questions.")

    # Check if vectorstore is ready
    if get_vectorstore(session_id) is None:
        st.info("⬆️ Please upload a PDF to start.")
        return

    # Display previous chat messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show chat input at bottom
    user_input = st.chat_input("Ask something about the paper...")
    if user_input:
        # Save and display user input
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Get response from chatbot
        answer = run_paper_chatbot(user_input, session_id)

        # Save and display assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)


