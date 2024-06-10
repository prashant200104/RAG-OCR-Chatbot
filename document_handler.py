import streamlit as st
from handling_images import save_images_to_pdf

openai_api_key = st.secrets["OPENAI_API_KEY"]

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
            #text_content = extract_text_from_text_pdf(file)
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
        [file.getvalue() for file in image_pdf_files], image_pdf_filenames, openai_api_key st.secrets["OPENAI_API_KEY"] )

    # Process text PDFs
    from brain_text import get_index_for_text_pdf
    text_vectordbs = get_index_for_text_pdf(
        [file.getvalue() for file in text_pdf], text_pdf_file_names, openai_api_key = st.secrets["OPENAI_API_KEY"] )
    

    return image_vectordbs + text_vectordbs
    


