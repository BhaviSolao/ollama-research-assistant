**Ollama-Based Research Assistant (RAG)**



I built this project to better understand how Retrieval-Augmented Generation (RAG)

can improve the reliability of large language models when answering questions

over documents.



The assistant runs fully locally using Ollama and does not rely on any external

APIs. This made it a good learning project for exploring privacy-preserving and

offline LLM workflows.



* What This Project Does

\- Allows users to ask questions over a set of documents

\- Retrieves only the most relevant document chunks using vector similarity

\- Generates answers strictly based on the retrieved context

\- Returns a safe response when no relevant information is found



* Tools Used

\- Ollama for local LLM inference

\- LangChain for building the RAG pipeline

\- FAISS for vector similarity search

\- Python



* How It Works 

1\. Documents are split into smaller chunks

2\. Each chunk is converted into an embedding

3\. Relevant chunks are retrieved using FAISS

4\. The language model generates an answer using only the retrieved content



* How to Run

1\. Install dependencies:



