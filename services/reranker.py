from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, question, documents, top_n=3):

        if not documents:
            return []

        pairs = [
            [question, doc.page_content]
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for doc, score in ranked[:top_n]:

            results.append({
                "document": doc,
                "score": float(score)
            })

        return results