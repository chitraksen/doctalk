from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from pyfiglet import figlet_format

from doctalk.core import *
from doctalk.config import Config

console = Console()


def mainMenu():
    # Create the title
    title = "[bold magenta]Main Menu[/bold magenta]"

    # Create a table to list the menu options
    table = Table(show_header=False, box=None)
    table.add_column("Options", justify="left", style="cyan")

    # Adding menu options
    table.add_row("[green]1.[/green] Search a directory.")
    table.add_row("[green]2.[/green] Query a file.")
    table.add_row("[green]3.[/green] Query a directory.")
    table.add_row("[green]4.[/green] Save a directory index.")
    table.add_row("[green]5.[/green] Have a chat!")
    table.add_row("[green]Q.[/green] Quit.")

    # Create a panel with the title and table
    menu_panel = Panel(Align.center(table), title=title, border_style="bright_blue")

    # Print the menu
    console.print(menu_panel)


def processInput() -> bool:
    console.print("\nInput choice: ", style="green", end="")
    choice = input()

    match choice:
        case "1":
            dirSearch()
        case "2":
            fileQuery()
        case "3":
            dirQuery()
        case "4":
            dirIndex()
        case "5":
            chat()
        case "q":
            console.print("Exiting application. Goodbye!\n", style="bold red")
            return False
        case "Q":
            console.print("Exiting application. Goodbye!\n", style="bold red")
            return False
        case _:
            print("Unrecognised input. Please try again.")

    return True


def main():
    # Title graphic
    console.print(figlet_format("DocTalk", font="larry3d"), style="bold cyan")

    # Initialize configuration
    global config
    config = Config()

    # Main application loop
    while True:
        mainMenu()
        if not processInput():  # returns false if user has chosen to exit
            break
