# Standard library imports
import json
import logging
from logging import Logger
import os
import time
from typing import Any, List, Literal

# Third-party imports
import requests
import streamlit as st
from dotenv import load_dotenv
import openai

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "data", ".env")
load_dotenv(dotenv_path)

# Global variables
news_api_key = os.getenv("NEWS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)
model = "gpt-3.5-turbo-16k"


def configure_logger(logger_name: str, log_file_path: str) -> Logger:
    """
    Configure a logger with the given name and file path.

    Parameters:
    - logger_name (str): The name of the logger
    - log_file_path (str): Path to the log file

    Returns:
    - logger (Logger): The configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


log_files = {
    "api_logger": "api_log.txt",
    "assistant_logger": "assistant_log.txt",
    "run_steps_logger": "run_steps_log.txt",
}

loggers = {}

for logger_name, log_file_name in log_files.items():
    log_file_path = os.path.join(os.path.dirname(__file__), "..", "logs", log_file_name)
    loggers[logger_name] = configure_logger(logger_name, log_file_path)


def fetch_data(topic: str) -> dict | None:
    """
    Fetches news data from News API for a given topic.

    Parameters:
    - topic (str): The topic to search for.

    Returns:
    - response (dict| None): The response from the API or None if an error occurred
    """
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    )
    try:
        response = requests.get(url)
        loggers["api_logger"].info(f"Fetching data from {url}")
        return response.json()
    except requests.exceptions.RequestException as e:
        loggers["api_logger"].error(f"Error fetching data: {e}")
        return None


def extract_articles(data) -> List[str] | None:
    """Extracts article data from News API response.

    Parameters:
    - data (dict): The response data from News API.

    Returns:
    - articles (List[str] | None): A list of article strings if successful or None if the API response status is not 'ok'.
    """

    status = data["status"]
    if status == "ok":
        loggers["api_logger"].info(f"News API returned status: {status}")
        return data["articles"]
    else:
        loggers["api_logger"].error(f"News API returned status: {status}")
        return []


def format_article(article) -> str:
    """
    Formats a news article as a string.

    Parameters:
    - article (dict): A news article object

    Returns:
    - formatted_article (str): The formatted news article
    """

    source = article["source"]["name"]
    author = article["author"]
    title = article["title"]
    description = article["description"]
    url = article["url"]
    content = article["content"]

    return f"""
        Title: {title},
        Author: {author},
        Source: {source},
        Description: {description},
        URL: {url},
    """


def get_news(topic: str) -> List[str] | None:
    """
    Gets the list of formatted news articles for a given topic.

    Parameters:
    - topic (str): The topic to search for.

    Returns:
    - formatted_articles (List[str] | None): The formatted news articles as a list of strings or None if no articles were found.
    """
    data = fetch_data(topic)
    if data is None:
        return None

    articles = extract_articles(data)
    if not articles:
        return []

    formatted_articles = [format_article(article) for article in articles]
    return formatted_articles


class AssistantManager:
    """
    Manages an OpenAI Assistant to have conversations and call functions.

    Creates an assistant and thread, sends messages to the thread, runs the
    assistant, calls functions based on the assistant's requests, waits for
    completion, and retrieves the summary.
    """

    assistant_id = "asst_RqIr4bdZKP49ZPFllrMWX23z"
    thread_id = "thread_Jxw1Gk6MybNkg2eAjtrk0ipY"

    def __init__(self, model: str = model) -> None:
        """
        Initialize the assistant manager.

        Parameters:
        - model (str): The model to use for the assistant.
        """

        self.client = client
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None

        # Check for existing assistant and thread
        if AssistantManager.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(
                assistant_id=AssistantManager.assistant_id
            )
            loggers["assistant_logger"].info(
                f"Found existing assistant with ID: {AssistantManager.assistant_id}"
            )
        if AssistantManager.thread_id:
            self.thread = self.client.beta.threads.retrieve(
                thread_id=AssistantManager.thread_id
            )
            loggers["assistant_logger"].info(
                f"Found existing thread with ID: {AssistantManager.thread_id}"
            )

    def create_assistant(self, name: str, instructions: str, tools: list) -> None:
        """
        Create an OpenAI Assistant.

        Parameters:
        - name (str): The name of the Assistant.
        - instructions (str): Instructions for the Assistant.
        - tools (list): The Tools to use for the Assistant.
        """
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                model=self.model,
                name=name,
                instructions=instructions,
                tools=tools,
            )
            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            loggers["assistant_logger"].info(
                f"Created new assistant with ID: {self.assistant.id}"
            )

    def create_thread(self) -> None:
        """
        Create a new Thread for the Assistant.
        """
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id
            self.thread = thread_obj
            loggers["assistant_logger"].info(
                f"Created new thread with ID: {self.thread.id}"
            )

    def add_message_to_thread(self, role: Literal["user"], content: str) -> None:
        """
        Add a message to the Assistant's Thread.

        Parameters:
        - role (str): The role of the message.
        - content (str): The content of the message.
        """
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content,
            )

    def run_assistant(self, instructions: str) -> None:
        """
        Run the Assistant.

        Parameters:
        - instructions (str): Instructions for the Assistant.
        """
        if self.assistant and self.thread:
            self.run = self.client.beta.threads.runs.create(
                assistant_id=self.assistant.id,
                thread_id=self.thread.id,
                instructions=instructions,
            )

    def process_message(self):
        """
        Process the latest message from the Assistant.
        """
        if self.thread:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id,
            )
            summary = []

            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)
            self.summary = "\n".join(summary)
            print(f"Summary: {role.capitalize()}: {response}")

    def call_required_functions(self, required_actions: dict[str, Any]) -> None:
        """
        Call the required functions.

        Parameters:
        - required_actions (dict): The required actions.
        """
        if not self.run:
            return
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            function_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if function_name == "get_news":
                output = get_news(topic=arguments["topic"])
                print(f"News on {arguments['topic']}: {output}")
                final_str = ""
                for item in output:
                    final_str += "".join(item)

                tool_outputs.append(
                    {
                        "tool_call_id": action["id"],
                        "output": final_str,
                    }
                )
            else:
                raise ValueError(f"Function {function_name} not found.")

            print("Submitting outputs back to Assistant...")
            self.client.beta.threads.runs.submit_tool_outputs(
                run_id=self.run.id,
                thread_id=self.thread.id,
                tool_outputs=tool_outputs,
            )

    def wait_for_completion(self):
        """
        Wait for the Assistant to complete.
        """
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id,
                )
                print(f"RUN STATUS: {run_status.model_dump_json(indent=4)}")

                if run_status.status == "completed":
                    self.process_message()
                    break
                elif run_status.status == "requires_action":
                    print("FUNCTION CALLING NOW...")
                    self.call_required_functions(
                        required_actions=run_status.required_action.submit_tool_outputs.model_dump()
                    )

    def get_summary(self) -> str | None:
        """
        Get the summary.

        Returns:
        - summary (str | None): The summary.
        """
        return self.summary

    def run_steps(self):
        if self.run and self.thread:
            run_steps = self.client.beta.threads.runs.steps.list(
                run_id=self.run.id, thread_id=self.thread.id
            )
            loggers["run_steps_logger"].info("Run Steps:")
            for step in run_steps.data:
                loggers["run_steps_logger"].info(f"{step}")


def main():
    # Create AssistantManager
    manager = AssistantManager()

    # Streamlit interface
    st.set_page_config(page_title="News Summarizer")
    st.title("News Summarizer")

    with st.form(key="user_input_form"):
        topic = st.text_input("Enter topic: ")
        submit_button = st.form_submit_button(label="Run Assistant")

        if submit_button:
            manager.create_assistant(
                name="News Summarizer",
                instructions="You are a personal article summarizer Assistant who knows how to take a list of article's titles and descriptions and then write a short summary of all the news articles",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_news",
                            "description": "Gets the list of formatted news articles for a given topic.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "topic": {
                                        "type": "string",
                                        "description": "The topic to search for, e.g. Artificial Intelligence",
                                    },
                                },
                                "required": ["topic"],
                            },
                        },
                    },
                ],
            )

            # Create Thread
            manager.create_thread()

            # Add the Message and run the Assistant
            manager.add_message_to_thread(
                role="user", content=f"Summarize the news on this topic: {topic}"
            )
            manager.run_assistant(instructions="Summarize the news.")

            # Wait for completion and process Messages.
            manager.wait_for_completion()
            summary = manager.get_summary()

            # Write summary to page
            st.write(summary)


if __name__ == "__main__":
    main()
