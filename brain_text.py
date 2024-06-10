import re
from io import BytesIO, StringIO
from typing import Tuple, List
import os
from dotenv import load_dotenv
import openai

from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import faiss

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import tabula

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_text_with_layout(file_obj):
    output_string = StringIO()
    laparams = LAParams(detect_vertical=True, all_texts=True)
    
    if isinstance(file_obj, BytesIO):
        file_obj.seek(0)
        extract_text_to_fp(file_obj, output_string, laparams=laparams, output_type='text')
    else:
        with open(file_obj, 'rb') as f:
            extract_text_to_fp(f, output_string, laparams=laparams, output_type='text')
    
    text_content = output_string.getvalue()
    output_string.close()
    
    return text_content

def extract_tables_from_pdf(file_obj):
    if isinstance(file_obj, BytesIO):
        file_obj.seek(0)
        tables = tabula.read_pdf(file_obj, pages='all', multiple_tables=True)
    else:
        tables = tabula.read_pdf(file_obj, pages='all', multiple_tables=True)
    return tables

def extract_text_from_text_pdf(file_obj):
    text_content = extract_text_with_layout(file_obj)
    #tables = extract_tables_from_pdf(file_obj)
    
    processed_data = {
        'text': text_content,
        #'tables': [table for table in tables]
    }
    
    if isinstance(processed_data, dict):
        processed_data = str(processed_data)
    
    return processed_data

def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    text = extract_text_from_text_pdf(file)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return [text], filename

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
            chunk_doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            chunk_doc.metadata["source"] = f"{chunk_doc.metadata['page']}-{chunk_doc.metadata['chunk']}"
            chunk_doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(chunk_doc)
    return doc_chunks

def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return index

def get_index_for_text_pdf(pdf_files, pdf_names, openai_api_key=None):
    if openai_api_key is None:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        docs = text_to_docs(text, filename)
        index = docs_to_index(docs, openai_api_key)
        documents.append(index)
    return documents

