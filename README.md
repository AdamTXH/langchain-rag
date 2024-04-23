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

With ingestion (optional):
```bash
python main.py -D "database/chromadb/flex_neo" -i -d "/home/adamtay/computex/documents/2024 Intel Flex Engineering Malaysia - NEO_20240228.pdf"
```
- -i, --ingest
- -d, --document_path <path_to_document>

With langfuse tracing:
```bash
python main.py -D "database/chromadb/flex_neo" -t
```

# Langfuse setup with persistant postgresql storage
Langfuse requires a persistent Postgres database to store its state.

### 1. Pull postgres image
```bash
docker pull postgres
docker run -d --name my_postgres -p 5432:5432 --network=host -e POSTGRES_PASSWORD=pass123 postgres
docker exec -it my_postgres bash
```

### 2. Create database (inside container)
```bash
psql -h localhost -U postgres
CREATE DATABASE langfuse;
```
- \l to verify 

### 3. Deploy langfuse
```bash
docker pull langfuse/langfuse:2
docker run --name langfuse \
-e DATABASE_URL=postgresql://postgres:pass123@localhost:5432/langfuse \
-e NEXTAUTH_URL=http://localhost:3000 \
-e NEXTAUTH_SECRET=mysecret \
-e SALT=mysalt \
-e HOSTNAME=0.0.0.0 \
-p 3000:3000 \
--net=host \
-a STDOUT \
langfuse/langfuse
```

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
