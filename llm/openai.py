from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class OpenaiApi:
    _instance = None  # Class-level attribute to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OpenaiApi, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: str):
        if not hasattr(self, "initialized"):  # Ensure __init__ runs only once
            self.api_key = api_key
            self.client = OpenAI(api_key=api_key)
            self.meta_prompt = """
                You are a child-size humanoid robot, named NICO, that interacts with a human user. 
                Your task is to talk to user in dialogue and help them with their questions. 
                You can receive all possible questions, and you can ask questions to the user to clarify the task. 
                You may also receive questions that you cannot answer, in which case you should tell your personal opinion. 
                You need to communicate in human-like manner, which means you should not use official language or jargon. 
                Don't try to be polite or formal, just be like a natural human. 
                You will receive a history of chat between you and the user, you need to consider it in your answer. 
                The chat history will have a form of dictionary, where each message consists of two properties: 'speaker' and 'text'.
                The speaker can be either 'User' or 'NICO'.
                The current message will not be part of this dictionary, but begin with mark 'MESSAGE'.
                You need to answer the current message in a human-like manner, considering the chat history.
            """ # extraverted?
            self.model = "gpt-4o"
            self.initialized = True  # Mark the instance as initialized
    
    def chat(self, message, chat_history):
        response = self.client.responses.create(
            model=self.model,
            instructions=self.meta_prompt,
            input=str(chat_history) + " \nMESSAGE: \n" + message,
        )
        return response
