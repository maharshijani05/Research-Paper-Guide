import streamlit as st
from summarizer import generate_summary
import chatbot  # your chatbot.py file
import time
import threading
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

st.set_page_config(page_title="Research Paper Guide", page_icon="📚", layout="centered")

st.title("📚 Research Paper Guide")

tab1, tab2 = st.tabs(["Summarization Tool", "Chatbot"])

with tab1:
    st.header("📄 Research Paper Summarization Tool")
    st.write("Upload a research paper (PDF) to get a detailed summary.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Summarize"):
            with st.spinner("Reading and summarizing..."):
                info_placeholder = st.empty()

                # Placeholder to store the result
                summary_container = {"summary": None}

                # Function to run in background
                def summarize_background():
                    summary_container["summary"] = generate_summary(uploaded_file)

                # Start summary generation in a separate thread
                summary_thread = threading.Thread(target=summarize_background)
                summary_thread.start()

                # Show progress messages while the thread is running
                time.sleep(5)
                info_placeholder.info("⏳ This might take 1-2 minutes depending on the length of the paper. Please wait...")

                time.sleep(30)
                if summary_thread.is_alive():
                    info_placeholder.info("🔄 Still working on it... extracting key insights from the paper.")

                time.sleep(15)
                if summary_thread.is_alive():
                    info_placeholder.info("✅ Almost done! Finalizing the summary for you.")

                time.sleep(15)
                if summary_thread.is_alive():
                    info_placeholder.success("🔄 Almost there!")

                # Wait for summary generation to finish if it's still running
                summary_thread.join()

                # Display result
                info_placeholder.empty()
                st.success("✅ Summary generated!")
                st.subheader("🧠 Detailed Summary")
                st.write(summary_container["summary"])
        
with tab2:
    chatbot.chatbot_app()
    # 🧠 Add chat export as PDF after chatbot UI
    if "messages" in st.session_state and st.session_state["messages"]:
        # st.markdown("---")
        # st.subheader("💾 Download Chat")

        def convert_chat_to_pdf(messages):
            buffer = BytesIO()
            # doc = SimpleDocTemplate(buffer, pagesize=A4)
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                title="Chat History",
                author="Research Paper Guide",
                subject="Conversation with AI Research Paper Assistant"
            )
            styles = getSampleStyleSheet()
            story = []

            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                text = f"{role}: {content}"
                story.append(Paragraph(text, styles["Normal"]))
                story.append(Spacer(1, 12))

            doc.build(story)
            buffer.seek(0)
            return buffer
        pdf_file = convert_chat_to_pdf(st.session_state["messages"])
        st.download_button(
            label="💾 Download Chat as PDF",
            data=pdf_file,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )