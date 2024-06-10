import re
import os
import io
import tempfile
from typing import Tuple, List

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

import PyPDF2
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import streamlit as st
import openai

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

from brain import get_index_for_pdf

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set the title for the Streamlit app
st.title("RAG-OCR Enhanced Chatbot")

prompt_template = """
    You are a helpful Assistant who answers users' questions based on multiple contexts given to you.
    Keep the answer correct and to the point. Try to answer from context first.
    Try answering in proper order and proper indentation; do not output in paragraphs much, use bullet points and all.
    If you did not get anything related to the query, print "Did not get any Related Information".
    The evidence is the context of the PDF extract with metadata.
    Only give a response and do not mention the source, page, or filename unless the user asks for it.
    The PDF content is:
    {pdf_extract}
"""

def save_images_to_pdf(images):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf_name = '_'.join(image_file.name for image_file in images) + '.pdf'

    for image in images:
        img = Image.open(image)
        img_width, img_height = img.size
        scale = min(letter[0] / img_width, letter[1] / img_height)
        img_width *= scale
        img_height *= scale
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        image_reader = ImageReader(img_buffer)
        c.drawImage(image_reader, 0, letter[1] - img_height, width=img_width, height=img_height)
        c.showPage()

    c.save()
    pdf_buffer.seek(0)
    return pdf_name, pdf_buffer

def initialize_session_state():
    if 'pdf_files' not in st.session_state:
        st.session_state.pdf_files = []

    if 'image_files' not in st.session_state:
        st.session_state.image_files = []

    if 'text_pdf_files' not in st.session_state:
        st.session_state.text_pdf_files = []

    if 'show_pdfs' not in st.session_state:
        st.session_state.show_pdfs = False

def handle_file_uploads():
    text_pdf_files = st.file_uploader("Upload Text PDF(s)", type="pdf", accept_multiple_files=True, key="text_pdf_upload")
    uploaded_pdf_files = st.file_uploader("Scanned/Handwritten PDF(s)", type="pdf", accept_multiple_files=True)
    image_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg", "gif"], accept_multiple_files=True, key="image_upload")

    if uploaded_pdf_files is not None:
        for file in uploaded_pdf_files:
            st.session_state.pdf_files.append((file.name, file))

    if image_files is not None:
        st.session_state.image_files.extend(image_files)
        if st.session_state.image_files:
            pdf_name, pdf_buffer = save_images_to_pdf(st.session_state.image_files)
            st.session_state.pdf_files.append((pdf_name, pdf_buffer))
            st.session_state.image_files = []
            st.session_state.show_pdfs = True

    if text_pdf_files is not None:
        for file in text_pdf_files:
            st.session_state.text_pdf_files.append((file.name, file))

    if 'vectordbs' not in st.session_state and (st.session_state.pdf_files or st.session_state.text_pdf_files):
        pdf_file_names = [name for name, _ in st.session_state.pdf_files]
        pdf_buffers = [buffer for _, buffer in st.session_state.pdf_files]
        text_pdf_file_names = [name for name, _ in st.session_state.text_pdf_files]
        text_pdf = [text for _, text in st.session_state.text_pdf_files]

        # Store document names for later use
        st.session_state.document_names = pdf_file_names + text_pdf_file_names

        st.session_state["vectordbs"] = create_vectordb(pdf_buffers, pdf_file_names, text_pdf, text_pdf_file_names)

@st.cache_resource
def create_vectordb(image_pdf_files, image_pdf_filenames, text_pdf, text_pdf_file_names):
    from brain import get_index_for_pdf

    # Process image PDFs
    image_vectordbs = get_index_for_pdf(
        [file.getvalue() for file in image_pdf_files], image_pdf_filenames, openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Process text PDFs
    from brain_text import get_index_for_text_pdf
    text_vectordbs = get_index_for_text_pdf(
        [file.getvalue() for file in text_pdf], text_pdf_file_names, openai_api_key=st.secrets["OPENAI_API_KEY"])

    return image_vectordbs + text_vectordbs

def initialize_prompt():
    if 'prompt' not in st.session_state:
        st.session_state.prompt = [{"role": "system", "content": "none"}]

def display_chat_history():
    for message in st.session_state.prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

def perform_similarity_search(vectordbs, question):
    pdf_extracts = []
    for vectordb in vectordbs:
        search_results = vectordb.similarity_search(question, k=10)
        pdf_extracts.append("\n".join([result.page_content for result in search_results]))
    return pdf_extracts

def generate_initial_responses(pdf_extracts, question, document_names):
    combined_responses = []
    for extract, doc_name in zip(pdf_extracts, document_names):
        individual_prompt = prompt_template.format(pdf_extract=extract)
        response = []
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": individual_prompt}, {"role": "user", "content": question}],
                temperature=0.6,
                stream=True
            )
            for chunk in completion:
                if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                    text = chunk['choices'][0]['delta']['content']
                    response.append(text)
            combined_responses.append((doc_name, "".join(response).strip()))
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return combined_responses

def refine_combined_response(combined_response_text, question):
    formatted_prompt = f"""
    I have gathered the following information from multiple sources in response to the question: "{question}"

    {combined_response_text}

    Please refine and improve the answer by making it more coherent and comprehensive and please do not repeat anything and output it in proper order.
    """
    final_response = []
    try:
        refinement = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.6,
            stream=True
        )
        for chunk in refinement:
            if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                text = chunk['choices'][0]['delta']['content']
                final_response.append(text)
    except Exception as e:
        st.error(f"An error occurred during refinement: {e}")
    return "".join(final_response).strip()

def handle_user_input(question):
    vectordbs = st.session_state.get("vectordbs", None)
    document_names = st.session_state.get("document_names", [])

    if not vectordbs:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    pdf_extracts = perform_similarity_search(vectordbs, question)
    combined_responses = generate_initial_responses(pdf_extracts, question, document_names)

    # Display individual document responses
    for doc_name, response in combined_responses:
        with st.chat_message("assistant"):
            st.write(f"From Document \"{doc_name}\" I received the following answer:")
            st.write(response)

    # Combine responses for final refinement
    combined_response_text = "\n\n".join([response for _, response in combined_responses])
    final_result = refine_combined_response(combined_response_text, question)

    st.session_state.prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        botmsg = st.empty()
        botmsg.write(final_result)

    st.session_state.prompt.append({"role": "assistant", "content": final_result})

def main():
    initialize_session_state()
    initialize_prompt()
    handle_file_uploads()
    
    # Print document names for testing
    st.write("Documents array:", st.session_state.get("document_names", []))
    
    display_chat_history()

    question = st.chat_input("Ask anything")
    if question:
        asyncio.run(handle_user_input(question))

if __name__ == "__main__":
    main()
