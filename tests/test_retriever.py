from app.rag.retriever import Retriever
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():

    print("=" * 60)
    print("NovelAI - Testing Retriever")
    print("=" * 60)

    retriever = Retriever()

    queries = [

        # Relevant
        "احكي لي كل ما تعرفه عن إيلثاريا",
        "ما قدرات إيلثاريا؟",
        "ما دور إيلثاريا في الخيانة؟",
        "من هو نيهالوم؟",
        "ما علاقتها بنيهلوم؟",
        "من هي إيلثاريا؟",
        "ما هي الاسلحة الموجوده؟",
        "ما هو الذكاء الاصطناعي؟", # Out-of-domain query
    ]

    for query in queries:

        print(f"\nQuery: {query}")
        print("=" * 60)

        results = retriever.search(query, top_k=7)

        if not results:
            print("\n🚨 Not Found / Out of Context")
            print("The system determined this question is unrelated to the novel or no context exists.")
            print("-" * 40)
            continue

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 40)

            print(f"Rerank Score : {result.get('rerank_score', 'N/A')}")
            print(f"Chroma Dist  : {result['distance']}")
            print(f"Metadata     : {result['metadata']}")

            print("\nText:")
            # Do not truncate to 500 characters, show the full chunk
            print(result["text"])

            print("-" * 40)


if __name__ == "__main__":
    main()
