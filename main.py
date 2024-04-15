import os
import random
from PyPDF2 import PdfReader
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
from bs4 import BeautifulSoup
from fpdf import FPDF
import requests
import dbts
import hdp


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

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
        {"role": "assistant", "content": "Upload some PDFs and ask me questions"}]

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

def scrape_paragraphs_with_keyword(urls, keyword):
    all_text = ""
    for url in urls:
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes

            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all paragraphs containing the keyword
            paragraphs = soup.find_all('p', string=lambda text: text and keyword.lower() in text.lower())

            # Extract text from paragraphs
            text = '\n\n'.join(paragraph.get_text() for paragraph in paragraphs)

            all_text += text.strip() + "\n\n"  # Add a newline between text from different pages
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to retrieve the webpage {url}: {e}")
        except Exception as e:
            st.error(f"Error parsing HTML from {url}: {e}")

    if all_text.strip():  # Check if there is any non-empty text
        return all_text.strip()
    else:
        return None  # Return None if no relevant content is found

def convert_to_pdf(text, output_file):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # Ensure text is properly encoded with utf-8
        text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=text)
        pdf.output(output_file)
        return True
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return False

def main():
    st.set_page_config(
        page_title="OCR-PDF Analyzer",
        page_icon="🩺"
    )

    # Sidebar for selecting options
    with st.sidebar:
        selected = option_menu(
            menu_title="Main menu",
            options=["Home", "Get PDF",  "Diabetes Prediction ML" ,"Heart Disease Prediction","About Us"],
            icons=["house-lock-fill", "file-pdf-fill", "cup", "person-lines-fill", "info-circle-fill"],
            default_index=0
        )

    # Home page
    if selected == "Home":
        st.title("A.I Medicial Assistant.📃")
        greetings = ["NAMASTE 🙏", "Hello! How can I assist you today?", "HOLA AMIGO ❤️", "Ready to work!"]
        selected_greeting = random.choice(greetings)
        st.write(selected_greeting)
        st.sidebar.button('Clear Recorded-data History', on_click=clear_chat_history)
        pdf_docs = st.file_uploader(
            "Upload your OCR-PDF Data and Click on the Extraction Button", accept_multiple_files=True, type=['pdf'])
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

        # Handle user input and display chatbot responses
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

    # Web Scraping page
    elif selected == "Get PDF":
        st.title("Web Scraping and PDF Generation")

        keyword = st.text_input("Enter keyword to search:")

        if st.button("Generate PDF"):
            urls = [
                "https://www.nhsinform.scot/illnesses-and-conditions/cancer/cancer-types-in-adults/brain-tumours/",
                "https://www.nhsinform.scot/illnesses-and-conditions/cancer/cancer-types-in-adults/bladder-cancer/",
                "https://www.nhsinform.scot/illnesses-and-conditions/brain-nerves-and-spinal-cord/chronic-pain",
                "https://www.nhsinform.scot/illnesses-and-conditions/skin-hair-and-nails/acne/",
                "https://www.nhsinform.scot/illnesses-and-conditions/muscle-bone-and-joints/conditions/arthritis/",
                "https://www.nhsinform.scot/illnesses-and-conditions/lungs-and-airways/asthma/",
                "https://www.nhsinform.scot/illnesses-and-conditions/mental-health/bipolar-disorder/",
                "https://www.nhsinform.scot/illnesses-and-conditions/cancer/cancer-types-in-adults/bowel-cancer/",
                "https://www.nhsinform.scot/illnesses-and-conditions/cancer/cancer-types-in-adults/breast-cancer-female/",
                "https://www.nhsinform.scot/illnesses-and-conditions/cancer/cancer-types-in-adults/cervical-cancer/",
                "https://www.nhsinform.scot/illnesses-and-conditions/immune-system/hiv/"
            ]
            output_file = keyword + ".pdf"
            text = scrape_paragraphs_with_keyword(urls, keyword)
            if text is not None:
                if convert_to_pdf(text, output_file):
                    st.success("PDF created successfully.")
                    st.write("Download the PDF file below:")
                    with open(output_file, "rb") as f:
                        pdf_bytes = f.read()
                        st.download_button(label="Download PDF", data=pdf_bytes, file_name=output_file,
                                           mime="application/pdf")
                else:
                    st.error("Failed to create PDF.")
            else:
                st.warning(f"No paragraphs containing the keyword '{keyword}' found on the webpage(s).")

    # About Us page
    elif selected == "About Us":
        st.title("About AIMS Project")
        st.header("Project Registration ID is ITE83")
        # st.write("Our project registration ID is .")
        st.write("Welcome to AIMS (A.I. Medical Assistant), a project designed to assist students and doctors in studying PDF materials.")
        st.header("Project Details")
        st.write("AIMS stands for A.I. Medical Assistant. It is a tool aimed at leveraging artificial intelligence to assist medical professionals and students in studying PDF materials efficiently.")
        st.header("Team Members")
        st.write("We are a team of four dedicated individuals working on the AIMS project:")
        st.write("- Abhishek")
        st.write("- Itesh")
        st.write("- Nandini")
        st.write("- Kritika")
    elif selected =="Diabetes Prediction ML":
        dbts.selected()
    elif selected =="Heart Disease Prediction":
        hdp.selected()
        

        

if __name__ == "__main__":
    main()
