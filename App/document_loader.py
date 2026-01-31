from pathlib import Path


def load_document(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks


def load_and_chunk_documents(directory, chunk_size=300):
    doc_path = Path(directory)
    all_chunks = []
    
    for file in doc_path.glob('*.txt'):
        content = load_document(file)
        chunks = chunk_text(content, chunk_size)
        
        for chunk in chunks:
            all_chunks.append({
                'text': chunk,
                'source': file.name,
                'length': len(chunk)
            })
    
    print(f"Loaded {len(all_chunks)} chunks from {len(list(doc_path.glob('*.txt')))} documents")
    return all_chunks


def get_chunk_stats(chunks):
    if not chunks:
        return {}
    
    lengths = [c['length'] for c in chunks]
    return {
        'total_chunks': len(chunks),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }
