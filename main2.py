import os
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"

# for embeddings
os.environ["OPENAI_API_KEY"] = "sk-xxx"

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings

# Initialize Langfuse handler
from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    secret_key="sk-lf-",
    public_key="pk-lf-",
    host="http://localhost:3000"
)

def main():
    llm = AzureChatOpenAI(model="llm-rag-chatgpt35")

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Option 1
    # vectorstore = Chroma.from_documents(documents=splits, persist_directory="./vectorstore-data", embedding=HuggingFaceEmbeddings(model_name="thenlper/gte-large"))

    # Option 2: Supports remote
    vectorstore = Chroma.from_documents(documents=splits, persist_directory="./vectorstore-data", embedding=HuggingFaceHubEmbeddings(model="http://localhost:8180"))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(rag_chain.invoke("what is task decomposition?", config={"callbacks": [langfuse_handler]}))

    # cleanup
    vectorstore.delete_collection()

if __name__ == "__main__":
    main()
