from research_assistant import load_text, split_text, build_faiss, answer_question

print("Loading text...")  #loading the text file into memory
docs = load_text()

print("Splitting text...") #splitting the text into smaller chunks for better retrieval
chunks = split_text(docs)

print("Building FAISS index...")  # Creating the FAISS index using embeddings of the chunks
vectorstore = build_faiss(chunks)

# Main question loop
while True:
    question = input("Ask your question:\n> ")
    answer = answer_question(vectorstore, question)
    print("\nANSWER:\n", answer)  #printing the clean final answer


