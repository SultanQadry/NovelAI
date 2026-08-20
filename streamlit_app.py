import streamlit as st
import time

from app.rag.retriever import Retriever
from app.ai.generator import Generator, GeneratorConfig

# ---------------------------------------------------------
# Page configuration & Modern Styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="NovelAI Worldbuilder",
    page_icon="🌌",
    layout="wide",
)

# Inject Custom CSS for a beautiful, premium aesthetic
st.markdown("""
<style>
    /* Dark glassmorphism chat bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 5px;
        border: 1px solid rgba(255,255,255,0.1);
        background-color: rgba(15, 20, 25, 0.6) !important;
        backdrop-filter: blur(10px);
    }
    .stChatMessage:nth-child(even) {
        background-color: rgba(30, 40, 50, 0.4) !important;
        border: 1px solid rgba(135,185,255,0.15);
    }
    
    /* Make the title pop with gradient */
    h1 {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-size: 3rem !important;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Subtle subtitle */
    .stMarkdown p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load models only once
# ---------------------------------------------------------

@st.cache_resource
def load_system():
    retriever = Retriever()
    generator = Generator(
        GeneratorConfig(
            provider="gemini",
            model="gemini-3.6-flash",
            temperature=0.3,
            max_tokens=1024,
        )
    )
    return retriever, generator


try:
    retriever, generator = load_system()
except Exception as e:
    st.error(f"Failed to initialize systems. Have you set your API keys? Error: {e}")
    st.stop()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", use_column_width=True)
    st.markdown("## 🌌 System Status")
    st.success("🟢 Embedding Engine Online")
    st.success("🟢 ChromaDB Connected")
    st.success("🟢 Reranker Initialized")
    st.success("🟢 Google Gemini API Connected")
    
    st.divider()
    st.markdown("### 📚 About NovelAI")
    st.markdown(
        "Transform your fantasy worldbuilding notes into a living entity. "
        "NovelAI retrieves lore exactly when you need it and answers questions natively."
    )
    st.caption("Powered by ChromaDB, HuggingFace & Gemini")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.title("NovelAI")
st.markdown("<p style='text-align: center; color: #888;'>Chat interactively with your fantasy worldbuilding knowledge base.</p>", unsafe_allow_html=True)
st.divider()


# Display previous messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🌌"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Lore Sources"):
                for idx, src in enumerate(message["sources"]):
                    st.caption(f"**Chunk {idx+1}** | Score: `{src.get('rerank_score', 'N/A')}`")
                    st.text(src.get("text", "No text provided"))
                    st.divider()


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------

if query := st.chat_input("Ask something about your world's history, characters, or magic..."):

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    # Prepare chunks safely
    chunks = []
    with st.spinner("📚 Scouring the archives..."):
        try:
            chunks = retriever.search(query, top_k=7)
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            st.stop()


    # Gather History
    history = []
    for msg in st.session_state.messages:
        # Exclude internal UI objects, map to exactly what generator expects
        history.append({"role": msg["role"], "content": msg["content"]})


    # Stream Output
    answer = ""
    with st.chat_message("assistant", avatar="🌌"):
        response_placeholder = st.empty()
        
        try:
            for token in generator.generate_stream(query, chunks, history=history):
                answer += token
                # Simulated typing effect delay for visual appeal is handled natively by stream
                response_placeholder.markdown(answer + "▌")
                
            response_placeholder.markdown(answer)  # final render without cursor
        except Exception as e:
            st.error(f"Generation Failed: {e}")
            st.stop()

        # Display sources context in the immediate output
        if chunks:
            with st.expander("🔍 View Retrieved Lore Sources"):
                for idx, src in enumerate(chunks):
                    score = src.get('rerank_score', 'N/A')
                    st.caption(f"**Chunk {idx+1}** | Relevance Score: `{score}`")
                    st.text(src.get("text", "No text provided"))
                    st.divider()

    # Save immediately to session state
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": chunks  # Keep sources so we can re-render them later
        }
    )