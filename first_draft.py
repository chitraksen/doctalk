import os
import argparse
import requests
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def load_documents(directory: str) -> List:
    loader = DirectoryLoader(directory, glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vectorstore(documents: List):
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_documents(documents, embeddings)

class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }
        response = requests.post(GROQ_API_ENDPOINT, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class GroqConversationalRetrievalChain:
    def __init__(self, llm: GroqLLM, retriever, memory: ConversationBufferMemory):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history = self.memory.chat_memory.messages
        relevant_docs = self.retriever.get_relevant_documents(question)
        relevant_content = "\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer."},
            {"role": "user", "content": f"Context: {relevant_content}\n\nQuestion: {question}"}
        ]

        for message in chat_history:
            if isinstance(message, HumanMessage):
                messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"role": "assistant", "content": message.content})

        response = self.llm(messages)
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)

        return {"answer": response}

def create_chain(vectorstore):
    llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return GroqConversationalRetrievalChain(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def main():
    parser = argparse.ArgumentParser(description="RAG CLI App")
    parser.add_argument("directory", help="Directory containing documents to process")
    args = parser.parse_args()

    print("Loading and processing documents...")
    documents = load_documents(args.directory)
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(documents)
    
    print("Initializing chat chain...")
    chain = create_chain(vectorstore)
    
    print("Chat initialized. Type 'exit' to end the conversation.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        
        response = chain({"question": query})
        print("AI:", response['answer'])

if __name__ == "__main__":
    main()