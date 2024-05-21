import re
import os
from io import BytesIO
from typing import Tuple, List
import streamlit as st
import tempfile
import logging
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Load environment variables
load_dotenv()

# Set the TESSDATA_PREFIX environment variable to the current directory
tessdata_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["TESSDATA_PREFIX"] = tessdata_dir

# Debug: Print the TESSDATA_PREFIX value
print(f"TESSDATA_PREFIX set to: {os.environ['TESSDATA_PREFIX']}")

# Ensure the TESSDATA_PREFIX is set correctly
tessdata_path = os.path.join(tessdata_dir, "eng.traineddata")
assert os.path.exists(tessdata_path), \
    f"TESSDATA_PREFIX is not set correctly or eng.traineddata is missing. Expected at {tessdata_path}"

# Set other environment variables or configurations
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def pdf_to_images(pdf_file: BytesIO) -> List[str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
        
        # Save PDF file to a temporary path
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.read())
        
        logging.debug(f"Temporary PDF path: {temp_pdf_path}")
        
        # Convert PDF to images
        try:
            images = convert_from_path(temp_pdf_path)
            logging.debug(f"Generated {len(images)} images from PDF")
        except Exception as e:
            logging.error(f"Error generating images: {e}")
            raise
        
        # Save images to temporary directory and return paths
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i+1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
            logging.debug(f"Saved image {i+1} at {image_path}")
        
        return image_paths

def parse_pdf(pdf_file: BytesIO, filename: str) -> Tuple[List[str], str]:
    output = []
    logging.info(f"Parsing PDF: {filename}")
    images = pdf_to_images(pdf_file)
    # Process the images (e.g., perform OCR)
    for i, image_path in enumerate(images, start=1):
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

