import json
import os
from rich.console import Console

console = Console()


class Config:
    def __init__(
        self,
    ):
        self.properties = [
            "llm_name",
            "llm_is_api",
            "llm_api_key",
            "embed_name",
            "embed_is_api",
            "embed_api_key",
        ]
        self.load()

    def load(self):
        # Load configuration from the JSON file.
        # Construct the full path to the config.json file
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(parent_dir, "config.json")

        if os.path.exists(config_path):
            # Read config file if it exists
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                    for prop in self.properties:
                        if prop in data:
                            setattr(self, prop, data[prop])
            except:
                console.print("Error reading file!", style="bold red")
        else:
            self.create()
            # Create config.json if it doesn't exist

    def create(self):
        console.print(
            "Previous configuration not found. Please create a new one.",
            style="bold cyan",
        )
        console.print("Choose your LLM. Currently only support Mistral.")
        # TODO: add more options
        choices = {
            1: "open-mixtral-8x7b",
            2: "open-mixtral-8x22b",
            3: "mistral-medium",
            4: "open-mistral-nemo",
        }
        for choice, model in choices.items():
            print(f"{choice}. {model}")
        while True:
            try:
                input_choice = int(input())
            except:
                print("Choice not valid. Please try again.")
                continue
            match input_choice:
                case 1 | 2 | 3 | 4:
                    self.llm_name = choices[input_choice]
                    self.llm_is_api = True
                    self.llm_api_key = self.getNewAPI()
                    break
                case _:
                    print("Choice not valid. Please try again.")
        self.embed_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
        self.embed_is_api = ""
        self.embed_api_key = False
        self.save()

    def getNewAPI(self) -> str:
        while True:
            key = input("The model needs an API Key. Please enter your API key: ")
            if len(key.strip()) > 0:
                break
            else:
                print("Blank key recieved.")
        return key

    def save(self):
        # Save the current configuration to the config.json file
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(parent_dir, "config.json")
        data = {prop: getattr(self, prop) for prop in self.properties}
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        console.print("Configuration saved!", style="bold green")

    def __repr__(self):
        # Return dict representation of the configuration
        return {prop: getattr(self, prop) for prop in self.properties}

    def __str__(self):
        # Return a string to print the configuration
        return "\n".join(
            f"{prop.capitalize()}: {getattr(self, prop)}" for prop in self.properties
        )


# Example usage
if __name__ == "__main__":
    config = Config()
    print(config)
