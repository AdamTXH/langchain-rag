import streamlit as st
import os
import torch
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




from haystack import Document
import time
from mlc_llm import ChatModule, ChatConfig, ConvConfig
from mlc_llm.callback import StreamToStdout
from mlc_llm.callback import StreamIterator
from threading import Event, Thread


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
        device=ComponentDevice.from_str("cpu"),
        meta_fields_to_embed=["title"]
    )
    document_embedder.warm_up()
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

    generator = ChatModule(
            device=_args.llm_device,
            model=_args.llm_model,
            model_lib_path=_args.llm_lib_path,
        )

    rag = Pipeline()
    embedder = SentenceTransformersTextEmbedder(
                        #model="sentence-transformers/all-MiniLM-L6-v2", # accuracy drop
                        #model="thenlper/gte-large", # accuracy drop
                        model="BAAI/bge-base-en-v1.5",
                        #model="BAAI/bge-large-en-v1.5",
                        #model="Snowflake/snowflake-arctic-embed-m", # accuracy drop
                        #model="Alibaba-NLP/gte-base-en-v1.5", #need trust_remote_code=True
                        device=ComponentDevice.from_str("cpu")
                      )
    embedder.warm_up()
    rag.add_component("text_embedder", embedder)
    rag.add_component("retriever",
                      InMemoryEmbeddingRetriever(document_store=document_store,
                      #MilvusEmbeddingRetriever(document_store=document_store,
                                               top_k=5)
    )
    ranker = TransformersSimilarityRanker(
                          model="BAAI/bge-large-en-v1.5",
                          #model="thenlper/gte-large", # similarity score format not match
                          #model="mixedbread-ai/mxbai-embed-large-v1", # similarity score format not match
                      )
    ranker.warm_up()
    rag.add_component("ranker",ranker)
    rag.add_component("prompt_builder", prompt_builder)

    rag.connect("text_embedder", "retriever")
    rag.connect("retriever.documents", "ranker.documents")
    rag.connect("ranker.documents", "prompt_builder.documents")
    
    rag.draw("rag.png")
    if _args.path != "":
        indexing.run({"file_type_router": {"sources": [_args.path]}})
    return rag, indexing, generator


#@st.cache_data(show_spinner=False)
def query_streamer(_pipeline, _question, _generator):
    chat_prompt = _pipeline.run({
        "text_embedder": {"text": _question},
        "ranker": {"query": _question}, #needed for TransformersSimilarityRanker
        "prompt_builder": {"query": _question}
      }
    )
    prompt = chat_prompt['prompt_builder']['prompt']
 
    stream = StreamIterator(callback_interval=2)
    generation_thread = Thread(
        target=_generator.generate,
        kwargs={"prompt": f"[INST] {prompt} [/INST]",
                "progress_callback": stream},
    )
    generation_thread.start()

    for delta_message in stream:
        yield delta_message

    generation_thread.join()

