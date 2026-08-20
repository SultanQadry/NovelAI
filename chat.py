"""
chat.py

Ties your existing Retriever together with the new Generator to give
a complete, interactive RAG chat loop for NovelAI.

Run:
    python chat.py
"""

import sys

from app.rag.retriever import Retriever          # your existing, working retriever
from app.ai.generator import Generator, GeneratorConfig


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    retriever = Retriever()

    generator = Generator(
        GeneratorConfig(
            provider="gemini",
            model="gemini-3.6-flash",
            temperature=0.3,
            max_tokens=1024,
        )
    )

    history = []  # running conversation for multi-turn context

    print("=" * 60)
    print("NovelAI — chat with your worldbuilding docs (type 'exit' to quit)")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        # 1. Retrieve
        chunks = retriever.search(query, top_k=7)

                # 2. Generate with streaming
        print("\nNovelAI: ", end="", flush=True)

        answer_parts = []

        for chunk in generator.generate_stream(
            query,
            chunks,
            history=history,
        ):  
            print(chunk, end="", flush=True)
            answer_parts.append(chunk)

        answer = "".join(answer_parts)

        print()

        # 3. Track history
        history.append({
            "role": "user",
            "content": query,
        })

        history.append({
            "role": "assistant",
            "content": answer,
        })


if __name__ == "__main__":
    main()