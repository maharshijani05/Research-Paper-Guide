import streamlit as st
from summarizer import generate_summary

st.set_page_config(page_title="Research Paper Summarizer")

st.title("📄 Research Paper Summarization Tool")
st.write("Upload a research paper (PDF) to get a detailed summary.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Reading and summarizing..."):
        summary = generate_summary(uploaded_file)
        st.success("Summary generated!")
        st.subheader("🧠 Detailed Summary")
        st.write(summary)
