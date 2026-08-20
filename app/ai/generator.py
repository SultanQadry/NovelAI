"""
app/ai/generator.py

The "Generation" layer of the NovelAI RAG pipeline.

Takes a user query + the ranked context chunks returned by
Retriever.search(query), builds a grounded system prompt, sends it
to an LLM, and returns a clean conversational answer.

Supports three interchangeable backends so you can swap providers
without touching the rest of your pipeline:
    - "anthropic"  -> Claude models via the Anthropic API
    - "openai"     -> GPT models via the OpenAI API
    - "local"      -> Any OpenAI-compatible local server (Ollama, LM Studio, vLLM, etc.)

Install (pick what you need):
    pip install anthropic          # for provider="anthropic"
    pip install openai             # for provider="openai" or provider="local"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator
from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    provider: str = "gemini"  # "gemini" | "anthropic" | "openai" | "local"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.3
    max_tokens: int = 1024
    # Only used when provider="local" (e.g. Ollama's OpenAI-compatible endpoint)
    base_url: Optional[str] = "http://localhost:11434/v1"
    # override which env var holds the key
    api_key_env_var: Optional[str] = None


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

class Generator:
    """
    Wraps an LLM call with a novel-worldbuilding-aware system prompt that
    is grounded strictly in the chunks your Retriever returns.
    """

    DEFAULT_SYSTEM_TEMPLATE = (
        "You are a knowledgeable assistant for a fantasy novel's worldbuilding "
        "bible. You answer questions ONLY using the CONTEXT provided below, which "
        "was retrieved from the author's own worldbuilding documents (characters, "
        "weapons, magic systems, locations, etc.).\n\n"
        "Rules:\n"
        "1. Base your answer strictly on the CONTEXT. Do not invent details that "
        "aren't supported by it.\n"
        "2. If the CONTEXT does not contain enough information to answer, say so "
        "plainly instead of guessing.\n"
        "3. Write in a natural, conversational tone — you're a lore assistant, "
        "not a search engine dumping raw text.\n"
        "4. When useful, mention which document/category a fact came from.\n\n"
        "CONTEXT:\n{context}"
    )

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self._client = self._build_client()

    # ---------------------------------------------------------------- #
    # Client setup
    # ---------------------------------------------------------------- #

    def _build_client(self):
        provider = self.config.provider.lower()

        if provider == "gemini":
            from google import genai

            key_var = self.config.api_key_env_var or "GEMINI_API_KEY"
            api_key = os.environ.get(key_var)

            if not api_key:
                raise ValueError(
                    f"{key_var} was not found in environment variables."
                )

            return genai.Client(api_key=api_key)

        if provider == "anthropic":
            import anthropic
            key_var = self.config.api_key_env_var or "ANTHROPIC_API_KEY"
            return anthropic.Anthropic(api_key=os.environ.get(key_var))

        if provider == "openai":
            import openai
            key_var = self.config.api_key_env_var or "OPENAI_API_KEY"
            return openai.OpenAI(api_key=os.environ.get(key_var))

        if provider == "local":
            # Local models are almost always served behind an
            # OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, etc.)
            import openai
            return openai.OpenAI(
                base_url=self.config.base_url,
                api_key="not-needed",  # most local servers ignore this
            )

        raise ValueError(f"Unknown provider: {self.config.provider!r}")

    # ---------------------------------------------------------------- #
    # Context formatting
    # ---------------------------------------------------------------- #

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Turns the list of chunk dicts from Retriever.search() into a single
        numbered context block. Assumes each chunk dict has at least a
        "text" key; adapt the field names to match your actual schema.
        """
        if not chunks:
            return "(No relevant context was found for this query.)"

        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", "").strip()
            score = chunk.get("rerank_score")
            score_str = f" (relevance: {score:.2f})" if score is not None else ""
            blocks.append(f"[{i}]{score_str}\n{text}")

        return "\n\n".join(blocks)

    def _build_system_prompt(self, chunks: List[Dict]) -> str:
        return self.DEFAULT_SYSTEM_TEMPLATE.format(context=self._format_context(chunks))

    # ---------------------------------------------------------------- #
    # Generation
    # ---------------------------------------------------------------- #

    def generate(
        self,
        query: str,
        chunks: List[Dict],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Args:
            query: the user's latest question.
            chunks: the List[dict] returned by Retriever.search(query).
            history: optional prior turns as [{"role": "user"/"assistant", "content": ...}, ...]
                     for multi-turn chat.

        Returns:
            The model's answer as a plain string.
        """
        system_prompt = self._build_system_prompt(chunks)
        messages = (history or []) + [{"role": "user", "content": query}]

        provider = self.config.provider.lower()

        if provider == "anthropic":
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text
            
        if provider == "gemini":
            chat = self._client.chats.create(model=self.config.model)
            full_query = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            if history:
                for msg in history:
                    full_query += f"{msg['role'].upper()}: {msg['content']}\n\n"
            full_query += f"USER: {query}"
            
            response = chat.send_message(full_query)
            return response.text

        # openai + local both use the chat.completions schema
        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        query: str,
        chunks: List[Dict],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """
        Same as generate(), but yields text chunks as they arrive —
        useful for a live-typing chat UI.
        """
        system_prompt = self._build_system_prompt(chunks)
        messages = (history or []) + [{"role": "user", "content": query}]
        provider = self.config.provider.lower()

        if provider == "anthropic":
            with self._client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
            return

        if provider == "gemini":
            chat = self._client.chats.create(model=self.config.model)
            full_query = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            if history:
                for msg in history:
                    full_query += f"{msg['role'].upper()}: {msg['content']}\n\n"
            full_query += f"USER: {query}"
            
            stream = chat.send_message_stream(full_query)
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
            
        stream = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta
