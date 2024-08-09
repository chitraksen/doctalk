import os
from rich.console import Console
from rich.markdown import Markdown

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from halo import Halo

from doctalk.utils import *

console = Console()


class FileNameRetriever(VectorIndexRetriever):
    def _retrieve(self, query, **kwargs):
        results = super()._retrieve(query, **kwargs)
        for node in results:
            # Add file name to the node's metadata
            node.metadata["file_name"] = os.path.basename(node.metadata["file_path"])
        return results


# TODO: figure out return type - sense checks?
def createIndex(path: str):
    spinner = Halo(text="Creating index...", spinner="dots", color="white")
    spinner.start()
    if os.path.isfile(path):
        documents = SimpleDirectoryReader(input_files=[path]).load_data()
    elif os.path.isdir(path):
        documents = SimpleDirectoryReader(path).load_data()
    # TODO: handle file not found errors
    else:
        console.print("Error!", stlye="bold red")

    Settings.llm = getLLM()
    Settings.embed_model = getEmbeddingModel()

    # TODO: persist vec index to disk and read from there - checks to see if files changed?
    index = VectorStoreIndex.from_documents(documents)

    spinner.succeed("File(s) indexed.")
    return index


# Function to get the most relevant file name
def get_relevant_file(query_engine, query):
    response = query_engine.query(query)
    if response.source_nodes:
        return response.source_nodes[0].metadata["file_name"]
    return "No relevant file found"


def dirSearch():
    # TODO: get path as input - option to use last path
    index = createIndex("docs")
    console.print("Find files in your directory.\n", style="bold")

    # set up the query engine with the custom retriever
    retriever = FileNameRetriever(index=index, similarity_top_k=2)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    # querying directory to seach for files
    console.print("What file are you looking for?", style="bold cyan")
    query = input()
    relevant_file = get_relevant_file(query_engine, query)
    console.print(
        f"[bold magenta]\nThe most relevant file is:[/bold magenta] {relevant_file}\n"
    )


def fileQuery():
    # TODO: get path as input - option to use last path
    # TODO: make repititive, history-aware chat
    index = createIndex("docs/romeo_and_juliet.txt")
    console.print("Chat with your file!", style="bold")

    # create query engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    console.print("[bold cyan]\nUser:[/bold cyan] ", end="")
    input_query = input()
    response = query_engine.query(input_query)
    console.print("[bold magenta]Assisstant[/bold magenta]:", str(response), "\n")


def dirQuery():
    # TODO: get path as input - option to use last path
    # TODO: make repititive, history-aware chat
    index = createIndex("docs")
    console.print("Chat with your directory!", style="bold")

    # create query engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    console.print("[bold cyan]\nUser:[/bold cyan] ", end="")
    input_query = input()
    response = query_engine.query(input_query)
    console.print("[bold magenta]Assisstant[/bold magenta]:", str(response), "\n")


def chat():
    # Set the LLM in the global settings
    Settings.llm = getLLM()

    # Initialize the ChatEngine
    chat_engine = SimpleChatEngine.from_defaults()

    console.print("Chat initialized with chosen LLM.", style="bold")
    print("Type 'exit' to end the conversation.\n")

    chat_history = []

    while True:
        console.print("User: ", style="bold cyan", end="")
        user_input = input()

        if user_input.lower() == "exit":
            console.print("Exiting conversation.\n", style="bold red")
            break

        # Add user message to chat history
        chat_history.append(ChatMessage(role=MessageRole.USER, content=user_input))

        # Generate a response using the chat engine
        response = chat_engine.chat(user_input, chat_history=chat_history)

        # Add AI response to chat history
        chat_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=response.response)
        )

        # Convert reply to markdown and print
        console.print("Assisstant: ", style="bold magenta", end="")
        console.print(Markdown(response.response))
        print()
