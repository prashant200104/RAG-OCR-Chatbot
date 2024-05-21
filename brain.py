import re
import os
from io import BytesIO
from typing import Tuple, List
import streamlit as st
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the TESSDATA_PREFIX environment variable
tessdata_dir = "/usr/share/tessdata"
# os.environ["TESSDATA_PREFIX"] = tessdata_dir
#os.environ["TESSDATA_PREFIX"] = "/usr/share/tessdata"
sudo mkdir -p /usr/share/tessdata
sudo wget -P /usr/share/tessdata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata


# Debug: Print the TESSDATA_PREFIX value
print(f"TESSDATA_PREFIX set to: {os.environ['TESSDATA_PREFIX']}")

# Ensure the TESSDATA_PREFIX is set correctly
tessdata_path = os.path.join(tessdata_dir, "eng.traineddata")
assert os.path.exists(tessdata_path), \
    f"TESSDATA_PREFIX is not set correctly or eng.traineddata is missing. Expected at {tessdata_path}"

# Set other environment variables or configurations
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import faiss

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def pdf_to_images(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name

    images = convert_from_path(temp_pdf_path)
    os.unlink(temp_pdf_path)

    return images

def parse_pdf(pdf_file: BytesIO, filename: str) -> Tuple[List[str], str]:
    output = []
    images = pdf_to_images(pdf_file)
    # Process the images (e.g., perform OCR)
    for i, image in enumerate(images, start=1):
        image_path = f'image_{i}.jpg'  # Save each image with a unique name
        image.save(image_path)

        # Perform OCR on the image
        text = pytesseract.image_to_string(Image.open(image_path))

        print(f"Page {i} OCR result:")
        # Clean up extracted text
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(doc)
    return doc_chunks

def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key = st.secrets["OPENAI_API_KEY"]))
    return index

def get_index_for_pdf(pdf_files, pdf_names, openai_api_key):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents, openai_api_key)
    return index

