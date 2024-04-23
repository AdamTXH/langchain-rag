# langchain-rag
Simple tutorial for running rag in langchain.

# Start Text embedding server based on Hugging face TEI (Instruction based on Nvidia 2080Ti)
Mount volume to model's path
```bash
model=thenlper/gte-large
volume=/data/llm/hf-tei-data
docker run --gpus all --env HTTPS_PROXY=$https_proxy --env HTTP_PROXY=$http_proxy -p 8180:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:turing-0.6 --model-id $model
```

# Run RAG Chatbot pipeline
Run chatbot:
```bash
python main.py -D "database/chromadb/flex_neo"
```
- -D, --database_path <path_to_database> (Required)

With ingestion:
```bash
python main.py -D "database/chromadb/flex_neo" -i -d "/home/adamtay/computex/documents/2024 Intel Flex Engineering Malaysia - NEO_20240228.pdf"
```
- -i, --ingest
- -d, --document_path <path_to_document>

# Langfuse metrics dashboard
[Langfuse](https://github.com/langfuse/langfuse) is an open-source LLM engineering platform that helps teams collaboratively debug, analyze, and iterate on their LLM applications.

### Sample Demo
![Langfuse Demo Trace](./langfuse_demo1.png)

![Langfuse Demo Dashboard](./langfuse_demo2.png)

### For quick PoC (not for production use as db not persistent)
```bash
# Clone the Langfuse repository
git clone https://github.com/langfuse/langfuse.git
cd langfuse
 
# Start the server and database
docker compose up
```

### For production use (need to prepare Postgres DB)
Refer to the [docs](https://langfuse.com/docs/deployment/self-host)
