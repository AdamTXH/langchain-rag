# langchain-rag
Simple tutorial for running rag in langchain.

# Start Text embedding server based on Hugging face TEI
model=thenlper/gte-large<br>
volume=/data/llm/hf-tei-data<br>
docker run --gpus all --env HTTPS_PROXY=$https_proxy --env HTTP_PROXY=$http_proxy -p 8180:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:turing-0.6 --model-id $model<br>
