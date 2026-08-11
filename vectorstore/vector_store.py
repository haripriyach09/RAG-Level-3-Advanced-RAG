from langchain_chroma import Chroma

from embeddings.embedding import EmbeddingModel
from config import VECTOR_DB_PATH


class VectorStore:

    @staticmethod
    def create(documents):

        embeddings = EmbeddingModel.load_embeddings()

        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        return db

    @staticmethod
    def load():

        embeddings = EmbeddingModel.load_embeddings()

        db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )

        return db