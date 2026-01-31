def build_grounded_prompt(query, retrieved_docs, confidence):
    if not retrieved_docs:
        return None, "NO_CONTEXT"
    
    if confidence < 0.3:
        return None, "LOW_CONFIDENCE"
    
    context = "\n\n".join([
        f"[Source: {doc['source']}]\n{doc['text']}"
        for doc in retrieved_docs
    ])
    
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Only use information from the context below
2. If the context doesn't contain the answer, say "I don't have enough information to answer that"
3. Do not make up or infer information
4. Cite the source when answering

Context:
{context}

Question: {query}

Answer (based only on the context above):"""
    
    return prompt, "READY"


def should_refuse(confidence, num_results):
    if num_results == 0:
        return True, "No relevant documents found"
    
    if confidence < 0.2:
        return True, "Confidence too low to provide accurate answer"
    
    return False, None


def build_refusal_response(reason):
    responses = {
        "NO_CONTEXT": "I don't have any relevant information to answer this question. Could you rephrase or ask something else?",
        "LOW_CONFIDENCE": "I found some related information, but I'm not confident enough to provide an accurate answer. Could you be more specific?",
        "No relevant documents found": "I couldn't find relevant information in my knowledge base. Could you try asking differently?",
        "Confidence too low to provide accurate answer": "I'm not confident I can answer this accurately based on available information."
    }
    
    return responses.get(reason, "I cannot answer this question with the available information.")
