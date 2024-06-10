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
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS


from brain import get_index_for_pdf
from document_handler import initialize_session_state, handle_file_uploads

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]


# in app.py -> from libraries.py import *
