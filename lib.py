import re
import os
import io
import tempfile
from typing import Tuple, List

import PyPDF2

from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import pytesseract
import streamlit as st
import openai

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS


from brain import get_index_for_pdf
from document_handler import initialize_session_state, handle_file_uploads

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]


# in app.py -> from libraries.py import *

#																	Useless Function for now

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

