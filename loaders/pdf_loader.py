from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    @staticmethod
    def load_pdf(file_path):

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        return documents