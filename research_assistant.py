from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# This function loads the text file that we want to use for our research assistant
def load_text():
    loader = TextLoader("Frequent Itemsets.txt", autodetect_encoding=True)
    return loader.load()

# Here I am splitting the text into smaller chunks so the model can process it better
def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # each chunk is around 800 characters
        chunk_overlap=150 # overlap so context is not lost
    )
    return splitter.split_documents(docs)

# This function creates embeddings for each chunk and stores them in a FAISS index
def build_faiss(chunks):
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # using a small but good embedding model
    )
    return FAISS.from_documents(chunks, embed)

# This function takes the user's question and finds the most relevant chunks
def answer_question(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    # joining all the retrieved text to pass into the LLM
    context = "\n\n".join(d.page_content for d in docs)

    prompt = (
        "Use the context below to answer the question clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    # connecting to the local Ollama model (llama3)
    model = ChatOllama(
        model="llama3",
        base_url="http://localhost:11434"
    )

    reply = model.invoke(prompt)

    # taking only the actual answer text from the model's response
    return reply.content.strip()
