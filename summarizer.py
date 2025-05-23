from PyPDF2 import PdfReader
from transformers import pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain import PromptTemplate

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

summarizer_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=1024,num_beams=4,truncation=True,early_stopping=True,no_repeat_ngram_size=3)

llm = HuggingFacePipeline(pipeline=summarizer_pipeline)

prompt_template = PromptTemplate.from_template(
    "Summarize the following research paper text in detail:\n\n{text}"
)

# chain = LLMChain(llm=llm, prompt=prompt_template)
chain = prompt_template | llm

def chunk_text(text, max_tokens=800):
    """Split text into smaller chunks without cutting mid-sentence."""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_tokens:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    chunks.append(current_chunk.strip())
    return chunks


def generate_summary(file):
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)

    all_summaries = []
    for chunk in chunks:
        # Ensure chunk is not too long for input
        chunk = chunk[:1000]
        summary = chain.invoke({"text": chunk})
        all_summaries.append(summary)

    return "\n\n".join(all_summaries)
