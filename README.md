# Hallucination-Controlled RAG Chatbot

This project focuses on building a Retrieval-Augmented Generation (RAG) chatbot that prioritizes factual accuracy and controlled responses over free-form generation.

## What this project does
- Grounds all answers strictly in retrieved document context  
- Uses retrieval-aware prompts to prevent unsupported responses  
- Implements refusal and clarification logic when context is missing or insufficient  
- Reduces hallucinations by tuning chunk size and prompt context structure  

## Why I built this
One of the biggest challenges with LLM-based systems is hallucination. This project was built to explore practical techniques for making RAG systems more reliable, predictable, and safe for real-world use.

## Key design decisions
- Prompt-level grounding instead of post-processing filters  
- Explicit refusal behavior for low-confidence queries  
- Careful chunking and context formatting to improve retrieval quality  

## Tech stack
- Large Language Models (LLMs)  
- Embeddings and vector search  
- Prompt engineering for controlled generation  

## Notes
This project emphasizes correctness and trustworthiness over creativity, making it suitable for enterprise-style RAG applications where accuracy is critical.
