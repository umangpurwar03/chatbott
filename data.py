import streamlit as st
import os
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import requests
import torch 

# Check if GPU is available and set it up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:",device)

# embedding_path=r"vector_space\faiss"
def pdf_data_loader(data_path):
    """
    This function loads PDF documents from a specified directory,
    splits the text content of the documents into smaller chunks, 
    creates embeddings for the text using Hugging Face's Sentence Transformers, 
    and uses Faiss to create an index for efficient similarity search, 
    The Faiss index is then saved to a local folder for future use.

    Parameters:
    - data_path (str): The path of the data folder containing PDF documents to be processed.

    Returns:
    None
    """

    # Step 1: Load PDF documents from the specified directory using DirectoryLoader
    data_loader = DirectoryLoader(data_path,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)
    print("Loading data...")

    # Load the documents from PDF files
    documents = data_loader.load()

    print("Data loaded successfully.")

    return documents

def csv_data_loader(data_path):
    data_loader = DirectoryLoader(data_path, 
                             glob='**/*.csv',
                             loader_cls=CSVLoader)
        # Load the documents from PDF files
    documents = data_loader.load()

    return documents

def xlsx_data_loader(data_path):
    data_loader = DirectoryLoader(data_path, 
                             glob='**/*.xlsx',
                             loader_cls=UnstructuredExcelLoader)
        # Load the documents from PDF files
    documents = data_loader.load()

    return documents

def url_data_loader(url):
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    urls = []
    for link in soup.find_all('a'):
        urls.append(link.get('href'))

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

    urls = urls + urls1
    data_loader = UnstructuredURLLoader(urls=urls)
    documents = data_loader.load()

    return documents

    
def vector(documents,embedding_path):

    # Step 2: Split the text content of documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
    
    print("Splitting text into chunks...")

    # Split documents into chunks of text
    texts = text_splitter.split_documents(documents)

    print("Text split into chunks.")

    # Step 3: Create embeddings for the text using Hugging Face's Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                        model_kwargs={'device': 'cpu'})
    
    print("Creating text embeddings...")

    # Step 4: Use Faiss to create an index for the text data
    db = FAISS.from_documents(texts, embeddings)

    print("Text embeddings created and indexed.")

    # Step 5: Save the Faiss index to a local folder for future use
    db.save_local(r"vector_space\faiss")

    print("Faiss index saved locally.")


def main():
    st.title("Text Embedding and Indexing App")

    data_source = st.radio("Select Data Source", ("Directory", "URL", "Both"))
    
    if data_source in ("Directory", "Both"):
        data_directory = st.text_input("Enter the directory path:")
    
    if data_source in ("URL", "Both"):
        url = st.text_input("Enter the URL:")
    
    embedding_path = tempfile.mkdtemp()  # Temporary folder to store embeddings

    # Create a folder to store the list of files in the database
    database_files_folder = os.path.join(embedding_path, "database_files")
    os.makedirs(database_files_folder, exist_ok=True)

    if st.button("Create Vector DB"):
        if data_source == "Directory" and data_directory:
            documents = pdf_data_loader(data_directory)
            documents += csv_data_loader(data_directory)
            documents += xlsx_data_loader(data_directory)
        elif data_source == "URL" and url:
            documents = url_data_loader(url)
        elif data_source == "Both" and data_directory and url:
            documents = pdf_data_loader(data_directory)
            documents += csv_data_loader(data_directory)
            documents += xlsx_data_loader(data_directory)
            documents += url_data_loader(url)
        else:
            st.warning("Please provide a valid data source (Directory, URL, or Both) and the required input.")

        if documents:
            vector(documents, embedding_path)
            st.success("Vector DB created and saved successfully.")
        else:
            st.error("No documents found for vectorization.")
    
if __name__ == "__main__":
    main()
