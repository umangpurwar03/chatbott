import streamlit as st
import random
import time
import torch
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredExcelLoader
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
# Import necessary modules from your chat code
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.document_loaders import SeleniumURLLoader

# Define the path where you want to save the Faiss index
EMBEDDING_PATH = r'vector_space\faiss'

DB_FAISS_PATH = r'vector_space\faiss'

# Function to load data from local files (PDFs, CSVs)
def load_local_data(data_path):
    if data_path.endswith(".pdf"):
        # Load PDF documents from the specified directory using DirectoryLoader
        data_loader = DirectoryLoader(data_path,
                                      glob='*.pdf',
                                      loader_cls=PyPDFLoader)
        print("Loading PDF data...")
        # Load the documents from PDF files
        documents = data_loader.load()
        print("PDF data loaded successfully.")
    elif data_path.endswith(".csv"):
        # Load text data from CSV files
        data_loader = DirectoryLoader(data_path,
                                      glob='**/*.csv',
                                      loader_cls=CSVLoader)
        print("Loading CSV data...")
        # Load the documents from CSV files
        documents = data_loader.load()
        print("CSV data loaded successfully.")
    else:
        st.error("Unsupported file format. Please provide PDF or CSV files.")

    return documents

# Function to load data from URLs
def load_url_data(url):
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    urls = []
    for link in soup.find_all('a'):
        urls.append(link.get('href'))

    # print(urls)
    # print(len(urls))

    urls1 = []
    for url in urls:
        try:
            reqs = requests.get(url)
            if reqs.status_code == 200:
                soup = BeautifulSoup(reqs.text, 'html.parser')
                for link in soup.find_all('a'):
                    urls1.append(link.get('href'))
            else:
                print(f"Access Denied for URL: {url}")
        except Exception as e:
            print(f"Error occurred for URL: {url} - {str(e)}")

    # print(urls1)
    # print(len(urls1))

    urls = urls + urls1
    print(len(urls))
    print(urls)
    data_loader = SeleniumURLLoader(urls=urls)
    # data_loader = UnstructuredURLLoader(urls=urls)
    print("Loading data from URLs...")
    # Load the documents from URLs
    documents = data_loader.load()
    print("Data loaded successfully from URLs.")

    return documents

# Function to create embeddings and save the Faiss index
def vector(documents):
    # Step 2: Split the text content of documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)

    print("Splitting text into chunks...")

    # Split documents into chunks of text
    texts = text_splitter.split_documents(documents)
    print(texts)

    print("Text split into chunks.")

    # Step 3: Create embeddings for the text using Hugging Face's Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                       model_kwargs={'device': 'cpu'})

    print("Creating text embeddings...")

    # Step 4: Use Faiss to create an index for the text data
    db = FAISS.from_documents(texts, embeddings)

    print("Text embeddings created and indexed.")

    return db

# Function to load and process data based on user choice
# Function to load and process data based on user choice
def data_loading_page():
    st.title("Data Loading")

    # User selects data source
    source_choice = st.radio("Select Data Source:", ["Local Files", "URLs", "Both"])

    if source_choice == "Local Files" or source_choice == "Both":
        # Add code to load data from local files (PDFs, CSVs)
        st.write("Enter the path to your data folder:")
        data_path = st.text_input("Path:")
        if st.button("Load Local Data"):
            # Call the load_local_data function to load data from local files
            documents = load_local_data(data_path)
            st.success("Local data loaded successfully!")

            # Create embeddings and save the Faiss index
            faiss_index = vector(documents)
            faiss_index.save_local(EMBEDDING_PATH)
            st.success("Faiss index for local data saved locally.")

    if source_choice == "URLs" or source_choice == "Both":
        # Add code to load data from URLs
        st.write("Enter the URL to fetch data:")
        data_url = st.text_input("URL:")
        if st.button("Load URL Data"):
            # Call the load_url_data function to load data from URLs
            documents = load_url_data(data_url)
            st.success("URL data loaded successfully!")

            # Create embeddings and save the Faiss index
            faiss_index = vector(documents)
            faiss_index.save_local(EMBEDDING_PATH)
            st.success("Faiss index for URL data saved locally.")

    # Function to clear the embedding directory
    def clear_embedding_directory(embedding_path):
        try:
            # Check if the directory exists
            if os.path.exists(embedding_path):
                # Remove all files in the directory
                for filename in os.listdir(embedding_path):
                    file_path = os.path.join(embedding_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                st.success("Embedding directory cleared successfully.")
            else:
                st.warning("Embedding directory does not exist.")
        except Exception as e:
            st.error(f"An error occurred while clearing the embedding directory: {str(e)}")

    if st.button("Clear Embedding Directory"):
        clear_embedding_directory(EMBEDDING_PATH)


# Define the chatbot function
def chatbot_page():
    st.title("Chatbot")
    #     # Clear the chat history when the chatbot page is loaded
    # if "messages" in st.session_state:
    #     st.session_state.messages = []


    # Check if GPU is available and set it up
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define a template for the prompt
    prompt_template = """
            your job is to act as a customer assistant
            If user query is a greeting, greet the user with a friendly response.
            If user query is a general question, provide a helpful answer from the database.
            if site is ecommerce then react as a customer assistant
            if you dont know answer or get from database ask user to rephrase
            just give answer
            Context: {context}
            Question: {question}

            """

    # Define responses for different greetings and common questions
    greeting_responses = {
        "hello": "Hello! How can I assist you?",
        # "hi": "Hi there! How can I assist you today?",
        # "hey": "Hey! How can I help you?",
    }

    common_question_responses = {
        "how are you": "I'm just a computer program, so I don't have feelings, but I'm here to assist you. How can I help?",
        "what are you doing": "I'm here to assist you with any questions or tasks you have. How can I assist you today?",
        "who are you": "I'm a chatbot designed to assist with answering questions and tasks. How can I assist you?",
        # Add more common questions and responses here
    }

    # Function to set up the prompt template
    def set_prompt():
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        return prompt

    # Function to create a Retrieval QA chain
    def retrieval(llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        return qa_chain

    # Function to load the language model
    def load_llm():
        llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.9
        )
        return llm

    # Function to set up the QA bot
    def qa_bot():
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})
        # Load the FAISS database
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        # Load the language model
        llm = load_llm()
        # Set up the prompt
        qa_prompt = set_prompt()
        # Create the QA chain
        qa = retrieval(llm, qa_prompt, db)
        return qa

    # Function to get the response to a user query
    def get_response(query, qa):
        response = qa({'query': query})
        return response['result']

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_input := st.chat_input("You:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Simulate the assistant's response with thinking delay
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Check if the user input is a greeting
            user_input_lower = user_input.lower()
            for greeting in greeting_responses:
                if greeting in user_input_lower:
                    assistant_response = greeting_responses[greeting]
                    break
            else:
                # Check if the user input is a common question
                for question in common_question_responses:
                    if question in user_input_lower:
                        assistant_response = common_question_responses[question]
                        break
                else:
                    # If it's not a greeting or a common question, get a response from the QA bot
                    response = get_response(user_input, qa_bot())
                    assistant_response = response
                    print(assistant_response)

            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Define the Streamlit app
if __name__ == "__main__":
    # Create a sidebar for navigation between pages
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page:", ["Data Loading", "Chatbot"])

    if page == "Data Loading":
        data_loading_page()
    elif page == "Chatbot":
        chatbot_page()
            # Clear the chat history when the chatbot page is loaded
        if "messages" in st.session_state:
            st.session_state.messages = []


        # Add a callback to rerun the app when the selected page changes
    selected_page = st.session_state.page if "page" in st.session_state else "Data Loading"
    if page != selected_page:
        st.session_state.page = page
        st.experimental_rerun()

    # Clear the chat history when the chatbot page is loaded
    if "messages" in st.session_state:
        st.session_state.messages = []
