import streamlit as st
from summarizer import generate_summary
import chatbot  # your chatbot.py file

st.set_page_config(page_title="Research Paper Guide")

st.title("📚 Research Paper Guide")

tab1, tab2 = st.tabs(["Summarization Tool", "Chatbot"])

with tab1:
    st.header("📄 Research Paper Summarization Tool")
    st.write("Upload a research paper (PDF) to get a detailed summary.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Summarize"):
            with st.spinner("Reading and summarizing..."):
                summary = generate_summary(uploaded_file)
                st.success("Summary generated!")
                st.subheader("🧠 Detailed Summary")
                st.write(summary)
        
with tab2:
    chatbot.chatbot_app()