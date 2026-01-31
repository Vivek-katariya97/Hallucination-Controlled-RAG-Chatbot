from chatbot import RAGChatbot


def run_demo():
    print("="*60)
    print("Hallucination-Controlled RAG Chatbot Demo")
    print("="*60)
    
    chatbot = RAGChatbot('documents', chunk_size=300)
    
    print(f"\nLoaded {chatbot.get_stats()['total_chunks']} chunks from {chatbot.get_stats()['documents']} documents")
    
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is quantum computing?",
        "Tell me about neural networks"
    ]
    
    print("\n" + "="*60)
    print("Running Test Queries")
    print("="*60)
    
    for query in test_queries:
        response = chatbot.ask(query)
        
        print(f"\nAnswer: {response['answer'][:200]}...")
        print(f"Sources: {response['sources']}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Refused: {response['refused']}")
        print("-"*60)
    
    print("\n" + "="*60)
    print("Interactive Mode")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        response = chatbot.ask(query)
        print(f"\nBot: {response['answer']}\n")
        
        if response['sources']:
            print(f"Sources: {', '.join(response['sources'])}")
        print(f"Confidence: {response['confidence']:.2f}\n")


if __name__ == '__main__':
    run_demo()
