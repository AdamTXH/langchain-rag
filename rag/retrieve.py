from rag.utils import benchmark

from langchain_chroma import Chroma

# Returns a retriever object from the chromadb
def get_chroma_retriever(chromadb_path, embedding_model, search_type="similarity", k=5):
    db = Chroma(persist_directory=chromadb_path, embedding_function=embedding_model)
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k": 5})
    return retriever

# Retrieve contexts from the retriever
@benchmark
def retrieve(retriever, query):
    contexts = retriever.invoke(query)
    return contexts