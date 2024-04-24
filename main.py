from rag.generate import *
from rag.ingest import *
from rag.retrieve import *
import argparse
import time
import asyncio

import langfuse
from langfuse.callback import CallbackHandler

import time

class MyCallbackHandler(CallbackHandler):
    def on_llm_end(
        self,
        response,
        *,
        run_id,
        **kwargs,
    ):
        '''
        Modified to add support for first token latency and token counts
        '''
        # print(response)
        # response.llm_output = {
        # "usage":{
        #     "input_tokens": 100,
        #      "output_tokens": 1000
        # }
        #  }
                
        self.runs[run_id].update(completion_start_time=response.llm_output.pop("completion"))

        super().on_llm_end(
            response,
            run_id=run_id,
            **kwargs
        )
        
langfuse_handler = MyCallbackHandler(
    secret_key="sk-lf-e6e2901e-7de3-4f17-8931-3960ccc2712e",
    public_key="pk-lf-6405031c-d29e-4103-b2d3-955ee758c9db",
    host="http://localhost:3000"
)

def main(options):
    if "database_path" not in options:
        raise ValueError("Database path is required.")

    # Embedding model
    #embedding_model = get_embedding_model("http://localhost:8188")
    embedding_model = get_hf_embedding_model()
    # Ingest
    if options["ingest"] == True:
        if "document_filepath" not in options:
            raise ValueError("Document filepath is required for ingest.")
        print("Ingesting...")
        # Load documents
        docs = load_documents(args.document_filepath)
        # Chunk documents
        chunks = chunk_documents(doc=docs, chunk_size=2000, chunk_overlap=0)
        # Save chunks to chroma
        save_chunks_to_chroma(chunks, embedding_model, args.database_path)
        print("...Ingestion Completed")
    # Retriever
    retriever = get_chroma_retriever(args.database_path, embedding_model, k=1)

    # Rag Chain
    # Using mistral on llamacpp for now
    #llm = get_llm_llamacpp("/data/llm/models/mistral-7b-instruct-v0.2.FP16.gguf")
    llm = get_llm_ipex()
    #llm = get_llm_hf()
    prompt_template = get_prompt_template(
        # """
        # {question}
        # """
    )
    rag_chain = create_rag_chain(retriever, prompt_template, llm)

    # Run interaction
    print(options)
    print("\nChat with LLM")
    print("===============")
    try:
        while True:
            user_query = input("You: ")
            t1 = time.time()
            print("AI: ", end='', flush=True)
            response = stream_output(rag_chain, user_query, trace = options["trace"], callback_handler=langfuse_handler)
            t2 = time.time() - t1
            print(f"Took {t2*1000:.4f} ms.")

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")

    # Args - Ingestion
    parser.add_argument("-i", "--ingest", action="store_const", const=True, default=False, help="To enable data ingestion.")
    parser.add_argument("-D", "--database_path", type=str, help="Specify a path to save the vector database.")
    parser.add_argument("-d", "--document_filepath", type=str, help="Specify the filepath to the document for rag.")

    # Args - Tracing (LangFuse)
    parser.add_argument("-t", "--trace", action="store_const", const=True, default=False, help="To enable langfuse tracing.")

    args = parser.parse_args()

    # Store options in a dictionary
    options = {k: v for k, v in vars(args).items() if v is not None}

    main(options)

# docker run -d --name my_postgres -v my_dbdata:/home/adamtay/computex/database/postgresql/data -p 5432:5432 -e POSTGRES_PASSWORD=my_password postgres:15-alpine
# docker exec -it my_postgres psql -U postgres

# docker run --name langfuse \
# -e DATABASE_URL=postgresql://postgres:pass123@localhost:5432/langfuse \
# -e NEXTAUTH_URL=http://localhost:3000 \
# -e NEXTAUTH_SECRET=mysecret \
# -e SALT=mysalt \
# -e HOSTNAME=0.0.0.0 \
# -p 3000:3000 \
# --net=host \
# -a STDOUT \
# langfuse/langfuse
