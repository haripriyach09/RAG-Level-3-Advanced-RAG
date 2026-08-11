from langchain_groq import ChatGroq
from config import GROQ_API_KEY


class LLM:

    @staticmethod
    def load(model_name):

        return ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature=0
        )