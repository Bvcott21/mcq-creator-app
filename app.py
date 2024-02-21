from dotenv import load_dotenv
import openai

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


load_dotenv()

##### Load Documents #####

# Function too read documents 
def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

# Passing the directory to the load_docs function
directory = './Docs/'
documents = load_docs(directory)


##### Transformer Documents #####

# Split documents into smaller chunks

# Function to split documents into chunks
def split_docs(documents, chunk_size = 1000, chunk_overlap = 20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

##### Generate Embeddings #####

# OpenAI LLM for creating embeddings for documents/texts
# embeddings = OpenAIEmbeddings(model_name="ada")

# HuggingFace LLM for Creating Embeddings for documents/text
embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

# Testing embeddings model for a sample text

query_result = embeddings.embed_query('Hello buddy')

print(len(query_result))


