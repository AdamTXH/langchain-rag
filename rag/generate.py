from rag import retrieve
from rag.utils import benchmark

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Get the llm chat model from Azure
def get_llm_azure(deployment_name, max_tokens=2048, temperature=0.7):
    return AzureChatOpenAI(
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        azure_deployment=deployment_name,
        max_tokens=max_tokens,
        temperature=temperature
    )

# def get_llm_hf()

# Create prompt template
def get_prompt_template(template=None):
    if template is None:
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    return custom_rag_prompt

# Format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the rag chain
def create_rag_chain(retriever, custom_rag_prompt, llm):
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
    )
    return rag_chain