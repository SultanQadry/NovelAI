# NovelAI: Fantasy Worldbuilding RAG

Welcome to **NovelAI**, an interactive Retrieval-Augmented Generation (RAG) system designed to bring your fantasy worldbuilding notes to life. 

With NovelAI, you can chat directly with your character sheets, magic systems, lore documents, and maps. The AI uses advanced retrieval techniques to find exact chunks from your database and synthesizes accurate, conversational responses without ever hallucinating lore.

## 🚀 Features

- **Blazing Fast Embeddings**: Local `sentence-transformers` automatically encode your documents.
- **ChromaDB Vector Store**: A robust database that stores and searches your lore history instantaneously.
- **Cross-Encoder Reranking**: Industry-standard two-stage retrieval. `ChromaDB` acts as a fast initial filter, and a highly accurate Cross-Encoder re-ranks the top results for maximum precision.
- **Universal LLM Support**: Designed strictly with an agnostic layer, you can swap between Google Gemini, Anthropic Claude, OpenAI, or local alternatives (Ollama/LMStudio) with a single line of code!
- **Premium Chat Interface**: Built on Streamlit, the custom UI features dark glassmorphism, transparent chat bubbles, and instant source expanders. 

---

## 🛠️ Installation Requirements

### 1. Requirements

Ensure you have Python 3.10+ installed. It is highly recommended to create an isolated virtual environment (`.venv`) for this project so you don't conflict with other Python packages on your computer.

#### Create and activate a Virtual Environment:
**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**On Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install dependencies:
Once your environment is active, install everything you need:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a file named `.env` in the root directory. Inside, paste the following (depending on your choice of LLM):

```env
# Required for retrieving HuggingFace models without rate limiting
HF_TOKEN=your_huggingface_read_token_here

# LLM Providers (add whichever you plan on using)
GEMINI_API_KEY=your_gemini_key
# ANTHROPIC_API_KEY=your_anthropic_key
# OPENAI_API_KEY=your_openai_key
```
*(You can generate a free HF token at [Hugging Face Settings](https://huggingface.co/settings/tokens)).*

---

## 🎮 How to Use

### Terminal CLI Loop
For a simple command-line interface testing:

```bash
python chat.py
```

### Beautiful Web UI (Streamlit)
To launch the interactive and immersive web UI:

```bash
streamlit run streamlit_app.py
```
This UI will boot into a stunning environment where you can query your knowledge base and physically read exactly which "chunks" of lore the AI used to build its answer by clicking the `View Retrieved Lore Sources` expander.

---

## 🧠 Modifying the Config

You can adjust how smart/creative the AI is by tweaking the configuration in `streamlit_app.py` or `chat.py`. 

```python
GeneratorConfig(
    provider="gemini",            # options: "gemini", "anthropic", "openai", "local"
    model="gemini-3.6-flash",     # or "claude-3-5-sonnet-20240620", "gpt-4o", etc.
    temperature=0.3,              # 0.0 (Strict Lore) to 0.7 (Creative storytelling)
    max_tokens=1024,
)
```

Enjoy writing your novel!
