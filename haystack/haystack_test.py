import os
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

from typing import List, Optional, Dict, Any
from haystack import component

import time

@component
class MyMlcLLM:
    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        return {
            "replies": ["hello", "world"],
            "generation_meta": {
                "first_token_latency": 5,
                "input_tokens": 65,
                "output_tokens": 100
            }
        }

#### decorator to record latency and other useful metrics ####
def decorate_method(cls, method_name, decorator):
    original_method = getattr(cls, method_name)
    if hasattr(original_method, '__wrapped__'):  # Check if method is already decorated
        original_method.__wrapped__ = decorator(original_method.__wrapped__)
    else:
        decorated_method = decorator(original_method)
        setattr(cls, method_name, decorated_method)

def my_decorator2(func):
    def wrapper(*args, **kwargs):
        # langfuse start
        start_time = time.time()

        output = func(*args, **kwargs)

        if "generation_meta" in output:
            # langfuse send additional metrics for generation here
            print(output.pop("generation_meta"))

        # langfuse end
        time.sleep(2)
        end_time = time.time()
        print(f"Latency of {func.__qualname__}: {end_time-start_time}")

        return output
    return wrapper

def decorate_all_components(pipeline):
    #for node in rag_pipeline.graph.nodes:
    for node in pipeline:
        temp = rag_pipeline.get_component(node)
        print(type(temp))
        print(dir(temp))
        print("\n\n")

        decorate_method(type(temp), 'run', my_decorator2)

#### ------------------------------------------------------------ ####

# Set the environment variable OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = "Your OpenAI API Key"

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."), 
    Document(content="My name is Mark and I live in Berlin."), 
    Document(content="My name is Giorgio and I live in Rome.")
])

# Build a RAG pipeline
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
#llm = OpenAIGenerator()
llm = MyMlcLLM()

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Adding another decorator to normal_method dynamically
#decorate_method(MyMlcLLM, 'run', my_decorator2)

decorate_all_components(rag_pipeline.graph.nodes)

# Ask a question
question = "Who lives in Paris?"
results = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
)

print(results["llm"]["replies"])

