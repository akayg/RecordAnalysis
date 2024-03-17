import os
import random
from PyPDF2 import PdfReader
import pandas as pd
import subprocess
import pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # read all pdf files and return text
# def run_python_script(script_name):
#     subprocess.run(["streamlit", "run", script_name], check=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n


    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain



def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="OCR-PDF Analyzer",
        page_icon="🩺"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        selected = option_menu(
            menu_title= "Main menu",
            options=["Home","PDFs","About_us"],
            icons=["house-lock-fill","file-pdf-fill","person-lines-fill"],
            default_index=0
        )
    if selected=="PDFs":
       st.title("leading")
    elif selected=="About_us":
       st.title("About us page")
    elif selected =="Home":
        st.title("Info retrieval through AI learning Chatbot.📃")
        greetings = ["NAMASTE 🙏", "Hello! How can I assist you today?", "HOLA AMIGO ❤️","Ready to work!"]
        selected_greeting = random.choice(greetings)
        st.write(selected_greeting)
        st.sidebar.button('Clear Recorded-data History', on_click=clear_chat_history)
        pdf_docs = st.file_uploader(
            "Upload your OCR-PDF Data and Click on the Extraction Button", accept_multiple_files=True,type=['pdf'])
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
             {"role": "assistant", "content": "Upload some records and ask me questions"}]
        
        if st.button("EXTRACTION"):
         with st.spinner("Learning from data ..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")
        
        for message in st.session_state.messages:
         with st.chat_message(message["role"]):
            st.write(message["content"])

        if prompt := st.chat_input():
         st.session_state.messages.append({"role": "user", "content": prompt})
         with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
        if st.session_state.messages[-1]["role"] != "assistant":
         with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
         if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
