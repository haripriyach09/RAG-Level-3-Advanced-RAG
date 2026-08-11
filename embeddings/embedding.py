from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


class EmbeddingModel:

    @staticmethod
    def load_embeddings():

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )