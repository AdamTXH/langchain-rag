from rag import retrieve
from rag.utils import benchmark

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from ipex_llm.langchain.llms import TransformersLLM
import os

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_llm_hf():
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf

# Create an llm chat model with Azure API
def get_llm_azure(deployment_name, max_tokens=2048, temperature=0.7):
    return AzureChatOpenAI(
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        azure_deployment=deployment_name,
        max_tokens=max_tokens,
        temperature=temperature,

    )

# Create a llamacpp model
def get_llm_llamacpp(model_path, n_gpu_layers=999, temperature=0, max_tokens=2048, top_p=1, verbose=False, n_ctx=2048):
    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        verbose=verbose,
        n_ctx=n_ctx
    )

def get_llm_ipex():
    return  TransformersLLM.from_model_id(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={"temperature": 0, "max_length": 4096, "trust_remote_code": True},
            device_map='xpu'
        )

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

# Streaming the output to improve responsiveness
async def stream_output(rag_chain, query, trace=False, callback_handler=None):
    response = ""
    if trace:
        async for text in rag_chain.astream(query, config={"callbacks": [callback_handler]}):
            print(text, end="", flush=True)
            response += text 
    else:
        async for text in rag_chain.astream(query):
            print(text, end="", flush=True)
            response += text 
    return response