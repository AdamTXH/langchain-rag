# langchain-rag
Simple tutorial for running rag in langchain.

# Start Text embedding server based on Hugging face TEI
Mount volume to model's path
```bash
model=thenlper/gte-large
volume=/data/llm/hf-tei-data
docker run --gpus all --env HTTPS_PROXY=$https_proxy --env HTTP_PROXY=$http_proxy -p 8180:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:turing-0.6 --model-id $model
```
