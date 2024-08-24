import os
from rich.console import Console
from rich.markdown import Markdown

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
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
    Settings.llm = getLLM()
    Settings.embed_model = getEmbeddingModel()
    spinner = Halo(text="Creating index...", spinner="dots", color="white")
    spinner.start()
    cached_path = os.path.join(path, ".DTcache")
    if os.path.exists(cached_path):
        # Load index from cache
        storage_context = StorageContext.from_defaults(persist_dir=cached_path)
        index = load_index_from_storage(storage_context)
        spinner.succeed("File(s) indexed.")
        return index
    elif os.path.isfile(path):
        documents = SimpleDirectoryReader(input_files=[path]).load_data()
    elif os.path.isdir(path):
        documents = SimpleDirectoryReader(path).load_data()
    # TODO: handle file not found errors
    else:
        spinner.fail("Error!")
        console.print("Exiting application.", style="bold red")
        raise SystemExit()

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


def getDir() -> str:
    console.print("Enter directory path.", style="cyan")
    console.print(
        "If absolute path is note provided, path is navigated relative to current working directory.",
        style="cyan",
    )
    console.print(
        "If left blank, current working directory will be chosen.", style="cyan"
    )
    while True:
        input_path = input()
        if input_path.strip() == "":
            input_path = os.getcwd()
            break
        elif os.path.exists(input_path):
            break
        else:
            print("Input path is not vaild. Please try again.")
    return input_path


def getFile() -> str:
    console.print("Enter directory path.", style="cyan")
    console.print(
        "If absolute path is note provided, path is navigated relative to current working directory.",
        style="cyan",
    )
    while True:
        input_path = input()
        if os.path.exists(input_path):
            break
        else:
            print("Input path is not vaild. Please try again.")
    return input_path


def dirSearch():
    dir_path = getDir()
    index = createIndex(dir_path)
    console.print("\nFind files in your directory.", style="bold")

    # set up the query engine with the custom retriever
    retriever = FileNameRetriever(index=index, similarity_top_k=2)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    # querying directory to seach for files
    console.print("What file are you looking for?", style="cyan")
    query = input()
    relevant_file = get_relevant_file(query_engine, query)
    console.print(f"[magenta]\nThe most relevant file is:[/magenta] {relevant_file}\n")


def fileQuery():
    # TODO: make repititive, history-aware chat
    file_path = getFile()
    index = createIndex(file_path)
    console.print("Chat with your file!", style="bold")

    # create query engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    console.print("[bold cyan]\nUser:[/bold cyan] ", end="")
    input_query = input()
    response = query_engine.query(input_query)
    console.print("[bold magenta]Assisstant[/bold magenta]:", str(response), "\n")


def dirQuery():
    # TODO: make repititive, history-aware chat
    dir_path = getDir()
    index = createIndex(dir_path)
    console.print("Chat with your directory!", style="bold")

    # create query engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    console.print("[bold cyan]\nUser:[/bold cyan] ", end="")
    input_query = input()
    response = query_engine.query(input_query)
    console.print("[bold magenta]Assisstant[/bold magenta]:", str(response), "\n")


def dirIndex():
    dir_path = getDir()
    index = createIndex(dir_path)
    index.storage_context.persist(persist_dir=f"{dir_path}/.DTcache")
    console.print("Index saved.", style="green")


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


# testing
if __name__ == "__main__":
    dirSearch()
