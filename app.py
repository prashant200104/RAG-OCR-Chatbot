# Import necessary libraries
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import databutton as db
import streamlit as st
import io
from openai import OpenAI
from dotenv import load_dotenv
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
from brain import get_index_for_pdf
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

# For Images
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# Set the title for the Streamlit app
st.title("RAG-OCR enhanced Chatbot")

# Initialize or retrieve the list of PDFs and image files
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = []

if 'image_files' not in st.session_state:
    st.session_state.image_files = []

if 'show_pdfs' not in st.session_state:
    st.session_state.show_pdfs = False

# Cached function to create a vectordb for the provided PDF files
@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Creating vector database..."):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai_api_key=st.secrets["OPENAI_API_KEY"])
    return vectordb

# Upload PDF files using Streamlit's file uploader
uploaded_pdf_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
image_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg", "gif"], accept_multiple_files=True)

# Function to convert images to a single PDF
def save_images_to_pdf(images):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Concatenate the names of all image files to create a PDF name
    pdf_name = '_'.join(image_file.name for image_file in images) + '.pdf'

    for image in images:
        img = Image.open(image)
        img_width, img_height = img.size

        # Calculate the scale to fit the image in the letter size page
        scale = min(letter[0] / img_width, letter[1] / img_height)
        img_width *= scale
        img_height *= scale

        # Save the image to a BytesIO object
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Create an ImageReader object from the BytesIO object
        image_reader = ImageReader(img_buffer)

        # Draw the image on the PDF canvas
        c.drawImage(image_reader, 0, letter[1] - img_height, width=img_width, height=img_height)
        c.showPage()

    c.save()
    pdf_buffer.seek(0)
    return pdf_name, pdf_buffer

# Add the uploaded images to the session state
if image_files:
    st.session_state.image_files.extend(image_files)
    #st.write("Uploaded Images:")
    #for image_file in st.session_state.image_files:
        #st.image(image_file, caption=image_file.name, use_column_width=True)

    pdf_name, pdf_buffer = save_images_to_pdf(st.session_state.image_files)
    st.session_state.pdf_files.append((pdf_name, pdf_buffer))

    # Clear the image files list after conversion
    st.session_state.image_files = []

    # Display a message confirming the PDF has been created
    #st.success(f"PDF '{pdf_name}' has been created and added to the list.")
    st.session_state.show_pdfs = True

# Add the uploaded PDF files to the session state
if uploaded_pdf_files:
    for file in uploaded_pdf_files:
        st.session_state.pdf_files.append((file.name, file))
    #st.session_state.show_pdfs = True

# Create the vectordb only if it hasn't been created yet
if 'vectordb' not in st.session_state and st.session_state.pdf_files:
    pdf_file_names = [name for name, _ in st.session_state.pdf_files]
    pdf_buffers = [buffer for _, buffer in st.session_state.pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_buffers, pdf_file_names)

# Define the template for the chatbot prompt
prompt_template = """
    You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    Keep answer correct and to the point. Try to answer from context first.

    If you did not get any relent solution print Dit not get any relevent solution
    
    The evidence are the context of the pdf extract with metadata. 
    
    Only give response and do not mention source or page or filename. If user asks for it, then tell.
        
    The PDF content is:
    {pdf_extract}
"""

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=5)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call ChatGPT with streaming and display the response as it comes
    response = []
    result = ""
    for chunk in client.chat.completions.create(
        model="gpt-3.5-turbo", messages=prompt, stream=True, temperature=0.6):
        text = chunk.choices[0].delta.content

        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

# Display the list of generated PDFs
#if st.session_state.show_pdfs and st.session_state.pdf_files:
    #st.write("Generated PDFs:")
    #for pdf_name, _ in st.session_state.pdf_files:
        #st.write(f"PDF Name: {pdf_name}")
    #st.session_state.show_pdfs = False  # Reset the flag to prevent repeated printing
