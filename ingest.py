import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders.pdf_loader import PDFLoader
from vectorstore.vector_store import VectorStore


PDF_FOLDER = "pdfs"


def main():

    all_documents = []

    pdf_files = [
        file
        for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    print(f"📚 Found {len(pdf_files)} PDF files")

    for file in pdf_files:

        path = os.path.join(
            PDF_FOLDER,
            file
        )

        print(f"📄 Loading: {path}")

        documents = PDFLoader.load_pdf(path)

        for doc in documents:

            doc.metadata["source"] = file

        print(
            f"✅ Loaded {len(documents)} pages"
        )

        all_documents.extend(documents)

    print(
        f"📚 Total pages loaded: "
        f"{len(all_documents)}"
    )

    print("🔄 Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        all_documents
    )

    print(
        f"✅ Created {len(chunks)} chunks"
    )

    print("🧠 Creating embeddings...")

    VectorStore.create(chunks)

    print("💾 Vector database created")

    print(
        "\n🎉 Level 3 ingestion completed!"
    )


if __name__ == "__main__":
    main()