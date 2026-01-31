from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentRetriever:
    
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunk_texts = [c['text'] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
    
    def retrieve(self, query, top_k=3, threshold=0.1):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                results.append({
                    'text': self.chunks[idx]['text'],
                    'source': self.chunks[idx]['source'],
                    'score': float(score)
                })
        
        return results
    
    def get_confidence(self, results):
        if not results:
            return 0.0
        
        top_score = results[0]['score']
        
        if len(results) > 1:
            score_gap = results[0]['score'] - results[1]['score']
        else:
            score_gap = top_score
        
        confidence = min(top_score * 2 + score_gap, 1.0)
        return confidence
