from dotenv import load_dotenv, dotenv_values
import openai

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
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



### Pinecone - For Semantic Search ###

'''
Pinecone allows for dat ato be uploaded into a vector database and true semantic
search to be performed.

Not only is conversational data highly structured, but it can also be complex.
Vector search and vector databases allows for similariy searches.

We will initialize Pinecone and create a Pinecone Index by passing our documents,
embeddings model and mentioning the specific index which has to be used.

These databases index vectors for easy search and retrieval by comparing values 
and finding those that are the most similar to one another, making them ideal for
natural language processing and AI-driven applications.
'''
# pc = Pinecone()

# index = pc.from_documents(docs, embeddings, index_name = "mcq-creator")
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name="mcq-creator")



### Retrieve Answers ###

# This function will help us in fetching the top relevant documents from our 
# vector store - Pinecone

def get_similar_docs(query, k=2):
    similar_docs = docsearch.similarity_search(query, k=k)
    return similar_docs

'''
'load_qa_chain' Loads a chain that you can use to do QA over a set of documents

And we'll be using HuggingFace for the reasoning purpose.
'''

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

'''
BigScience Large Open-science Open-access Multilingual Language Model (BLOOM)
is a transformer-based large language model.

It was created by over 1000 AI researchers to provide a free language model for
everyone who wants to ty. Trained on around 366 billion tokens over March 
through July 2022, it is considered an alternative to OpenAI's GPT-3 with its 
176 billion parameters
'''
llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature": 1e-10})



'''
Different types of chain_type:

- map_reduce    -   It divides the texts into batches, processes each batch with
                    the question, and combines the answers to provide the final 
                    answer.
- refine        -   It divides the text into batches and refines the answer by 
                    sequentially processing each batch with the previous answer.
- map_rerank    -   It divides the text into batches, evaluates the quality of 
                    each answer from LLM, and selects the highest-scoring answers
                    from the batches to generate the final answer. These alternatives
                    help handle token limitations and improve the effectiveness 
                    of the question answering process.
'''
chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    relevant_docs = get_similar_docs(query)
    print(relevant_docs)
    response = chain.run(input_documents=relevant_docs, question=query)
    return response

# Passing question to the above created function
our_query = "How is Venezuelan economy?"
answer = get_answer(our_query)

### Structure the output ###

import re
import json

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(
        name = "question", 
        description = "Question generated from provided input text data."),
    ResponseSchema(
        name = "choices",
        description = "Available options for a multiple-choice question in comma separated"
    ),
    ResponseSchema(
        name = "answer",
        description = "Correct answer for the asked question."
    )
] 

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# This helps us fetch the instructions that langchain creates to fetch the 
# response in desired format
format_instructions = output_parser.get_format_instructions()
print("\n\nFormat instructions\n\n")
print(format_instructions)

chat_model = ChatOpenAI()

prompt = ChatPromptTemplate(
    messages = [
        HumanMessagePromptTemplate.from_template("""
            when a text input is given by the user, please generate a multiple
            choice question from it along with the correct answer.
            \n{format_instructions}\n{user_prompt}""")
    ],
    input_variables = ["user_prompt"],
    partial_variables = {"format_instructions": format_instructions}
)

final_query = prompt.format_prompt(user_prompt = answer)
print("\n\nFinal Query:\n\n")
print(final_query)

final_query.to_messages()

final_query_output = chat_model(final_query.to_messages())
print("\n\nFinal Query Output:\n\n")
print(final_query_output.content)

'''
While working with scenarios like above where we have to process multi-line
strings(separated by newline characters - '\n' ). In such situations, we use
re.DOTALL 
'''

# Extracting JSON data from Markdown text that we have
markdown_text = final_query_output.content
json_string = re.search(
    r'{(.*?)}',
    markdown_text,
    re.DOTALL
    ).group(1)

print("\n\nResponse in JSON format\n\n")
print(json_string)