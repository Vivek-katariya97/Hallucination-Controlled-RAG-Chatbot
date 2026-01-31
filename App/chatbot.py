from document_loader import load_and_chunk_documents
from retriever import DocumentRetriever
from prompt_builder import build_grounded_prompt, should_refuse, build_refusal_response


class RAGChatbot:
    
    def __init__(self, documents_dir, chunk_size=300):
        print("Loading documents...")
        self.chunks = load_and_chunk_documents(documents_dir, chunk_size)
        
        print("Initializing retriever...")
        self.retriever = DocumentRetriever(self.chunks)
        
        print("Chatbot ready!")
    
    def ask(self, query, top_k=3):
        print(f"\nQuery: {query}")
        
        results = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            print("Status: No relevant documents found")
            return {
                'answer': build_refusal_response("No relevant documents found"),
                'sources': [],
                'confidence': 0.0,
                'refused': True
            }
        
        confidence = self.retriever.get_confidence(results)
        print(f"Confidence: {confidence:.2f}")
        
        refuse, reason = should_refuse(confidence, len(results))
        
        if refuse:
            print(f"Status: Refused - {reason}")
            return {
                'answer': build_refusal_response(reason),
                'sources': [r['source'] for r in results],
                'confidence': confidence,
                'refused': True
            }
        
        prompt, status = build_grounded_prompt(query, results, confidence)
        
        if status != "READY":
            print(f"Status: {status}")
            return {
                'answer': build_refusal_response(status),
                'sources': [],
                'confidence': confidence,
                'refused': True
            }
        
        simulated_answer = self._generate_answer(query, results)
        
        print(f"Status: Answered (confidence: {confidence:.2f})")
        return {
            'answer': simulated_answer,
            'sources': [r['source'] for r in results],
            'confidence': confidence,
            'refused': False,
            'prompt': prompt
        }
    
    def _generate_answer(self, query, results):
        context_snippets = [r['text'][:200] for r in results]
        sources = list(set([r['source'] for r in results]))
        
        answer = f"Based on the available documents ({', '.join(sources)}), here's what I found:\n\n"
        answer += context_snippets[0] + "..."
        
        return answer
    
    def get_stats(self):
        return {
            'total_chunks': len(self.chunks),
            'documents': len(set([c['source'] for c in self.chunks]))
        }
