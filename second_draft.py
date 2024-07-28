import os
import requests
import json
import chardet
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Mapping, Any

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.llms import CompletionResponse, CustomLLM
from llama_index.core.extractors import SummaryExtractor
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def process_document(file_path):
    try:
        # Detect the file encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        # Read the file with the detected encoding
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()

        # Create metadata
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": os.path.splitext(file_path)[1],
            "file_size": os.path.getsize(file_path),
            "creation_date": os.path.getctime(file_path),
            "modification_date": os.path.getmtime(file_path),
            "encoding": encoding,
        }

        # Create a Document object
        doc = Document(text=content, metadata=metadata)

        return doc
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def load_documents(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    documents = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_document, file_path): file_path for file_path in file_paths}
        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                documents.append(result)

    return documents

def create_index(documents):
    '''
    prob do this bit locally using phi3 mini. doesn't have to be sota model, just get summaries and then worry about query LLM. can also inc chunk size to like 128k or something stupid.
    technique - https://openai.com/index/summarizing-books/
    '''

    # Initialize the LLM
    llm = Ollama(model="phi3:mini-128k", request_timeout=6000000.0)

    # Create a text splitter
    text_splitter = TokenTextSplitter(chunk_size=100000, chunk_overlap=1000)

    # Create a summary extractor
    summary_extractor = SummaryExtractor(
        llm=llm,
        summaries=["prev", "self"],
        summary_prompt=(
            "Please provide a concise summary of the following text, "
            "focusing on key information and definitions:"
        ),
    )

    # Create the index with the text splitter and summary extractor
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter, summary_extractor],
    )

    return index

def chat_with_documents():
    folder_path = input("Enter the path to the folder containing your documents: ")
    documents = load_documents(folder_path)
    index = create_index(documents)
    
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    query_engine = index.as_query_engine(llm=llm)

    print("Chat with your documents (type 'quit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = query_engine.query(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("Please set the GROQ_API_KEY environment variable.")
    else:
        chat_with_documents()