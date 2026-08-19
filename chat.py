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
            provider="anthropic",         # switch to "openai" or "local" as needed
            model="claude-sonnet-4-6",
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

        # 2. Generate
        answer = generator.generate(query, chunks, history=history)

        print(f"\nNovelAI: {answer}")

        # 3. Track history for follow-up questions
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()