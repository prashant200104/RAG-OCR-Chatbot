import openai
from lib import *
from brain import get_index_for_pdf
from document_handler import initialize_session_state
from document_handler import handle_file_uploads


# Set the title for the Streamlit app
st.title("RAG-OCR Enhanced Chatbot")

prompt_template = """
    You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    Keep answer correct and to the point. Try to answer from context first.

	if I ask any question, collect answers from all pdf's and then give me add of all answers.
	
	Give answer in very detail and in proper order.
    
    The evidence are the context of the pdf extract with metadata. 
    
    Only give response and do not mention source or page or filename. If user asks for it, then tell.
        
    The PDF content is:
    {pdf_extract}
"""

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
        search_results = vectordb.similarity_search(question, k=5)
        pdf_extracts.append("\n".join([result.page_content for result in search_results]))
    return pdf_extracts

def generate_initial_responses(pdf_extracts, question):
    combined_responses = []
    for extract in pdf_extracts:
        individual_prompt = prompt_template.format(pdf_extract=extract)
        response = []
        try:
            completion = openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": individual_prompt}, {"role": "user", "content": question}],
                temperature=0.6,
                stream=True
            )
            for chunk in completion:
                if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                    text = chunk['choices'][0]['delta']['content']
                    response.append(text)
            combined_responses.append("".join(response).strip())
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
        refinement = openai.ChatCompletion.acreate(
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
    if not vectordbs:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    pdf_extracts = perform_similarity_search(vectordbs, question)
    combined_responses = generate_initial_responses(pdf_extracts, question)
    combined_response_text = "\n\n".join(combined_responses)
    #final_result = refine_combined_response(combined_response_text, question)

    st.session_state.prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        botmsg = st.empty()
        botmsg.write(combined_response_text)

    st.session_state.prompt.append({"role": "assistant", "content": final_result})

def main():
    initialize_session_state()
    initialize_prompt()
    handle_file_uploads()
    display_chat_history()

    question = st.chat_input("Ask anything")
    if question:
        handle_user_input(question)

if __name__ == "__main__":
    main()
