import streamlit as st


enable_xpu = True
if enable_xpu:
    import ipex_llm # add ipex patch

import os
import torch
from utils.telemetry import enable_content_tracing, is_content_tracing_state_enabled

from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.rankers import TransformersSimilarityRanker, LostInTheMiddleRanker, MetaFieldRanker
#from milvus_haystack import MilvusDocumentStore
#from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
#from haystack_integrations.components.generators.ollama import OllamaGenerator

from hf_ipex import IpexHuggingFaceLocalChatGenerator
from haystack.dataclasses import ChatMessage

from haystack import Document
import time
# from mlc_llm import ChatModule, ChatConfig, ConvConfig
# from mlc_llm.callback import StreamToStdout
# from mlc_llm.callback import StreamIterator
from threading import Event, Thread
from queue import Queue
from typing import Optional


class XPUDevice(object):
    def __init__(self) -> None:
        self.first_device = self

    def to_torch_str(self) -> str:
        return "xpu"

    def to_torch(self) -> str:
        return "xpu"


class Callback:
    def __init__(self, callback_interval: int = 2, timeout: Optional[float] = None):
        self.delta_messages: Queue = Queue()
        self.callback_interval = callback_interval
        self.timeout = timeout

    def __call__(self, chunk, stopped: bool = False):
        if stopped:
            self.stopped_callback()
        else:
            delta = chunk.content
            self.delta_callback(delta)

    def delta_callback(self, delta_message: str):
        self.delta_messages.put(delta_message, timeout=self.timeout)

    def stopped_callback(self):
        self.delta_messages.put(None, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.delta_messages.get(timeout=self.timeout)
        if value:
            return value
        raise StopIteration()
    
class HfIpexCallback(Callback):
    def __call__(self, chunk: str, stopped: bool = False):
        if stopped:
            self.stopped_callback()
        else:
            self.delta_callback(chunk)


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def set_initial_state():
    set_state_if_absent("question", "Ask something here?")
    set_state_if_absent("results", None)


# cached to make index and models load only at start
@st.cache_resource(show_spinner=False)
def initialize_haystack_rag(_args):
    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine")

    """
    document_store = MilvusDocumentStore(
        connection_args={
            "host": "localhost",
            "port": "19530",
            "user": "",
            "password": "",
            "secure": False,
        },
        drop_old=True,
    )
    """

    file_type_router = FileTypeRouter(
        mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()
    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="sentence",
                                         split_length=2)

    document_embedder = SentenceTransformersDocumentEmbedder(
        #model="sentence-transformers/all-MiniLM-L6-v2", # accuracy drop
        #model="thenlper/gte-large", # accuracy drop
        model="BAAI/bge-base-en-v1.5",
        #model="BAAI/bge-large-en-v1.5",
        #model="Alibaba-NLP/gte-base-en-v1.5", #need trust_remote_code=True
        #model="Snowflake/snowflake-arctic-embed-m", # accuracy drop
        #device=ComponentDevice.from_str("xpu"),
        meta_fields_to_embed=["title"],
        normalize_embeddings=True
    )
    instruction = "Represent this sentence for searching relevant passages:"
    embedder = SentenceTransformersTextEmbedder(
                        #model="sentence-transformers/all-MiniLM-L6-v2", # accuracy drop
                        #model="thenlper/gte-large", # accuracy drop
                        model="BAAI/bge-base-en-v1.5",
                        #model="BAAI/bge-large-en-v1.5",
                        #model="Snowflake/snowflake-arctic-embed-m", # accuracy drop
                        #model="Alibaba-NLP/gte-base-en-v1.5", #need trust_remote_code=True
                        prefix=instruction,
                        normalize_embeddings=True,
                        #device=ComponentDevice.from_str("xpu")
                      )

    ranker = TransformersSimilarityRanker(
                          #model="BAAI/bge-large-en-v1.5",
                          model="BAAI/bge-reranker-base",
                          #model="thenlper/gte-large", # similarity score format not match
                          #model="mixedbread-ai/mxbai-embed-large-v1", # similarity score format not match,
                          #device=ComponentDevice.from_str("xpu")
                      )
    ranker.warm_up()
    if enable_xpu:
        document_embedder.device = XPUDevice()
        embedder.device = XPUDevice()
        ranker.model.to('xpu')
        ranker.device = XPUDevice()

    document_embedder.warm_up()
    embedder.warm_up()

    document_writer = DocumentWriter(document_store,
                                     policy=DuplicatePolicy.OVERWRITE)

    indexing = Pipeline()
    indexing.add_component(instance=file_type_router,
                           name="file_type_router")
    indexing.add_component(instance=text_file_converter,
                           name="text_file_converter")
    indexing.add_component(instance=markdown_converter,
                           name="markdown_converter")
    indexing.add_component(instance=pdf_converter,
                           name="pypdf_converter")
    indexing.add_component(instance=document_joiner,
                           name="document_joiner")
    indexing.add_component(instance=document_cleaner,
                           name="document_cleaner")
    indexing.add_component(instance=document_splitter,
                           name="document_splitter")
    indexing.add_component(instance=document_embedder,
                           name="document_embedder")
    indexing.add_component(instance=document_writer,
                           name="document_writer")

    indexing.connect("file_type_router.text/plain",
                     "text_file_converter.sources")
    indexing.connect("file_type_router.application/pdf",
                     "pypdf_converter.sources")
    indexing.connect("file_type_router.text/markdown",
                     "markdown_converter.sources")
    indexing.connect("text_file_converter", "document_joiner")
    indexing.connect("pypdf_converter", "document_joiner")
    indexing.connect("markdown_converter", "document_joiner")
    indexing.connect("document_joiner", "document_cleaner")
    indexing.connect("document_cleaner", "document_splitter")
    indexing.connect("document_splitter", "document_embedder")
    indexing.connect("document_embedder", "document_writer")
    indexing.draw("indexing.png")


    prompt_template = """<|system|>Using the information contained in the context, give a comprehensive answer to the question.
    If the answer is contained in the context, also report the source metadata if available.
    Please provide answer with full sentence, and don't end a sentence with colon.
    If the context is not relevant, please answer the question by using your own knowledge about the topic.</s>
    <|user|>
    Context:
      {% for doc in documents %}
      {{ doc.content }}
      {% endfor %};
      Question: {{query}}
      </s>
    <|assistant|>
    """

    prompt_builder = PromptBuilder(template=prompt_template)

    #streamer = Callback(callback_interval=2)
    # generator = OllamaGenerator(model="llama3:8b-instruct-q4_K_M",
    #                             url="http://localhost:11434/api/generate",
    #                             generation_kwargs={
    #                                 #"num_predict": 100,
    #                                 "temperature": 0.0,
    #                             },
    #                             streaming_callback=streamer
    #                             )

    streamer = HfIpexCallback()
    generator = IpexHuggingFaceLocalChatGenerator(model="mistralai/Mistral-7B-Instruct-v0.2", streaming_callback=streamer)

    rag = Pipeline()
    rag.add_component("text_embedder", embedder)
    rag.add_component("retriever",
                      InMemoryEmbeddingRetriever(document_store=document_store,
                      #MilvusEmbeddingRetriever(document_store=document_store,
                                               top_k=5)
    )

    rag.add_component("ranker",ranker)
    rag.add_component("prompt_builder", prompt_builder)

    rag.connect("text_embedder", "retriever")
    rag.connect("retriever.documents", "ranker.documents")
    rag.connect("ranker.documents", "prompt_builder.documents")
    
    rag.draw("rag.png")
    if _args.path != "":
        indexing.run({"file_type_router": {"sources": [_args.path]}})
    return rag, indexing, generator, streamer


#@st.cache_data(show_spinner=False)
def query_streamer(_pipeline, _question, _generator, _streamer):
    content_tracing_state = is_content_tracing_state_enabled()
    enable_content_tracing(True)
    start_time = time.time()
    chat_prompt = _pipeline.run({
        "text_embedder": {"text": _question},
        "ranker": {"query": _question}, #needed for TransformersSimilarityRanker
        "prompt_builder": {"query": _question}
      }
    )
    end_time = time.time()
    print(f"pipeline run time: {(end_time-start_time)*1000} ms.")
    prompt = chat_prompt['prompt_builder']['prompt']

    generation_thread = Thread(
        target=_generator.run,
        #kwargs={"prompt": f"[INST] {prompt} [/INST]"},
        kwargs={"messages": [ChatMessage.from_user(prompt)]},
    )
    generation_thread.start()

    for delta_message in _streamer:
        yield delta_message

    generation_thread.join()

    enable_content_tracing(content_tracing_state)


#@st.cache_data(show_spinner=False)
def process_documents(_indexing, _args):
    start_time = time.time()
    if not os.path.exists(_args.tmp_dir):
        os.mkdir(_args.tmp_dir)
    if st.session_state['file_uploader'] is not None:
        with st.spinner("Processing document ..."):
            doc_list = []
            for source_doc in st.session_state['file_uploader']:
                path = os.path.join(_args.tmp_dir, source_doc.name)
                if "files" not in st.session_state.keys() or path not in st.session_state.files:
                    if "files" not in st.session_state.keys():
                        print(f"no files key. add {path} to it")
                        st.session_state.files = [path]
                    else:
                        st.session_state.files.append(path)
                    if not os.path.exists(path):
                        with open(path, "wb") as f:
                            f.write(source_doc.getvalue())
                        print(f"write {path}")
                    doc_list.append(path)
                    print(f"add {path} to doc_list")
            if len(doc_list) > 0:
                print(f"adding {path} to database")
                _indexing.run(
                    {"file_type_router": {"sources": doc_list}})
        duration = time.time() - start_time
    print(f"it took {duration} to process document")


#@st.cache_data(show_spinner=False)
def update_documents(_question, _comment, _indexing, _args):
    start_time = time.time()
    content = f"For question {_question}, don't forget {_comment}"

    fname = f"feedback_{st.session_state.run_id}.txt"
    path = os.path.join(_args.tmp_dir, fname)
    with open(path, "w") as f:
        f.write(content)
    _indexing.run({"file_type_router": {"sources": [path]}})

    duration = time.time() - start_time
    print(f"**********for question {_question}, update with feedback '{_comment}'. It took {duration}s")

