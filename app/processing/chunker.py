import re

def chunk_text(text, chunk_size=300, overlap_sentences=2):
    """
    Splits text into chunks while preserving sentence boundaries and paragraphs.
    Uses sentence-based overlap instead of word-based to prevent cutting context mid-sentence.
    """
    # Split text into sentences using standard and Arabic punctuation, as well as newlines
    sentences = re.split(r'(?<=[.!?؟\n])\s+', text.strip())
    
    chunks = []
    current_sentences = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)
        
        # If appending the sentence exceeds chunk size (and we already have sentences)
        if current_length + sentence_len > chunk_size and current_sentences:
            # Save the current valid chunk
            chunks.append(" ".join(current_sentences))
            
            # Start a new chunk with the last 'overlap_sentences' to retain context cleanly
            overlap_list = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            
            current_sentences = overlap_list + [sentence]
            # Recalculate length safely
            current_length = sum(len(s.split()) for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_length += sentence_len
            
    # Add any remaining sentences as the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))
        
    return chunks
