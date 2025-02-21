import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

# Load models
model = SentenceTransformer('BAAI/bge-small-en-v1.5')  # Better accuracy
nlp = spacy.load("en_core_web_sm")  # For sentence splitting

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text.append((page_num, text))
    return full_text

def process_pdf(pdf_path):
    texts = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(texts)
    embeddings = get_embeddings(chunks)
    return chunks, embeddings


def clean_text(text):
    text = text.replace('\n', ' ').replace('\x0c', ' ')  # Clean PDF artifacts
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

def smart_sentence_split(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def split_into_chunks(texts, chunk_size=500, overlap=50):
    chunks = []
    for page_num, text in texts:
        sentences = smart_sentence_split(clean_text(text))
        current_chunk = []
        for sentence in sentences:
            if len(' '.join(current_chunk + [sentence])) > chunk_size:
                chunks.append({
                    "page": page_num,
                    "text": ' '.join(current_chunk)
                })
                current_chunk = current_chunk[-overlap:] + [sentence]  # Add overlap
            else:
                current_chunk.append(sentence)
        if current_chunk:
            chunks.append({"page": page_num, "text": ' '.join(current_chunk)})
    return chunks

def get_embeddings(chunks):
    """Generate embeddings for text chunks"""
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts)

def find_most_similar(prompt, chunks, embeddings, top_k=1):
    """Find top-k most similar chunks to prompt"""
    prompt_embedding = model.encode([prompt])
    similarities = np.dot(embeddings, prompt_embedding.T).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_indices]

def find_full_context(chunk, all_chunks, window_size=1):
    """Get surrounding chunks for context"""
    index = all_chunks.index(chunk)
    start = max(0, index - window_size)
    end = min(len(all_chunks), index + window_size + 1)
    return " [...] ".join([c["text"] for c in all_chunks[start:end]])